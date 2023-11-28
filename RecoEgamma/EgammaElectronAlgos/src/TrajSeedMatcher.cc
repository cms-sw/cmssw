#include "RecoEgamma/EgammaElectronAlgos/interface/TrajSeedMatcher.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "TrackingTools/TrajectoryState/interface/ftsFromVertexToPoint.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

constexpr float TrajSeedMatcher::kElectronMass_;

namespace {
  auto makeMatchingCuts(std::vector<edm::ParameterSet> const& cutsPSets) {
    std::vector<std::unique_ptr<TrajSeedMatcher::MatchingCuts> > matchingCuts;

    for (const auto& cutPSet : cutsPSets) {
      int version = cutPSet.getParameter<int>("version");
      switch (version) {
        case 1:
          matchingCuts.emplace_back(std::make_unique<TrajSeedMatcher::MatchingCutsV1>(cutPSet));
          break;
        case 2:
          matchingCuts.emplace_back(std::make_unique<TrajSeedMatcher::MatchingCutsV2>(cutPSet));
          break;
        default:
          throw cms::Exception("InvalidConfig") << " Error TrajSeedMatcher::TrajSeedMatcher pixel match cuts version "
                                                << version << " not recognised" << std::endl;
      }
    }

    return matchingCuts;
  }

  TrajSeedMatcher::SCHitMatch makeSCHitMatch(const GlobalPoint& vtxPos,
                                             const TrajectoryStateOnSurface& trajState,
                                             const TrackingRecHit& hit,
                                             float et,
                                             float eta,
                                             float phi,
                                             int charge,
                                             int nrClus) {
    EleRelPointPair pointPair(hit.globalPosition(), trajState.globalParameters().position(), vtxPos);
    float dRZ = hit.geographicalId().subdetId() == PixelSubdetector::PixelBarrel ? pointPair.dZ() : pointPair.dPerp();
    return {hit.geographicalId(), hit.globalPosition(), dRZ, pointPair.dPhi(), hit, et, eta, phi, charge, nrClus};
  }

  const std::vector<TrajSeedMatcher::MatchInfo> makeMatchInfoVector(
      std::vector<TrajSeedMatcher::SCHitMatch> const& posCharge,
      std::vector<TrajSeedMatcher::SCHitMatch> const& negCharge) {
    std::vector<TrajSeedMatcher::MatchInfo> matchInfos;
    size_t nrHitsMax = std::max(posCharge.size(), negCharge.size());
    for (size_t hitNr = 0; hitNr < nrHitsMax; hitNr++) {
      DetId detIdPos = hitNr < posCharge.size() ? posCharge[hitNr].detId : DetId(0);
      float dRZPos = hitNr < posCharge.size() ? posCharge[hitNr].dRZ : std::numeric_limits<float>::max();
      float dPhiPos = hitNr < posCharge.size() ? posCharge[hitNr].dPhi : std::numeric_limits<float>::max();

      DetId detIdNeg = hitNr < negCharge.size() ? negCharge[hitNr].detId : DetId(0);
      float dRZNeg = hitNr < negCharge.size() ? negCharge[hitNr].dRZ : std::numeric_limits<float>::max();
      float dPhiNeg = hitNr < negCharge.size() ? negCharge[hitNr].dPhi : std::numeric_limits<float>::max();

      if (detIdPos != detIdNeg && (detIdPos.rawId() != 0 && detIdNeg.rawId() != 0)) {
        cms::Exception("LogicError") << " error in " << __FILE__ << ", " << __LINE__
                                     << " hits to be combined have different detIDs, this should not be possible and "
                                        "nothing good will come of it";
      }
      DetId detId = detIdPos.rawId() != 0 ? detIdPos : detIdNeg;
      matchInfos.push_back({detId, dRZPos, dRZNeg, dPhiPos, dPhiNeg});
    }
    return matchInfos;
  }
};  // namespace

TrajSeedMatcher::Configuration::Configuration(const edm::ParameterSet& pset, edm::ConsumesCollector&& cc)
    : magFieldToken{cc.esConsumes()},
      paramMagFieldToken{cc.esConsumes(pset.getParameter<edm::ESInputTag>("paramMagField"))},
      navSchoolToken{cc.esConsumes(pset.getParameter<edm::ESInputTag>("navSchool"))},
      detLayerGeomToken{cc.esConsumes(pset.getParameter<edm::ESInputTag>("detLayerGeom"))},
      useRecoVertex{pset.getParameter<bool>("useRecoVertex")},
      enableHitSkipping{pset.getParameter<bool>("enableHitSkipping")},
      requireExactMatchCount{pset.getParameter<bool>("requireExactMatchCount")},
      useParamMagFieldIfDefined{pset.getParameter<bool>("useParamMagFieldIfDefined")},
      minNrHits{pset.getParameter<std::vector<unsigned int> >("minNrHits")},
      minNrHitsValidLayerBins{pset.getParameter<std::vector<int> >("minNrHitsValidLayerBins")},
      matchingCuts{makeMatchingCuts(pset.getParameter<std::vector<edm::ParameterSet> >("matchingCuts"))} {
  if (minNrHitsValidLayerBins.size() + 1 != minNrHits.size()) {
    throw cms::Exception("InvalidConfig")
        << " TrajSeedMatcher::TrajSeedMatcher minNrHitsValidLayerBins should be 1 less than minNrHits when its "
        << minNrHitsValidLayerBins.size() << " vs " << minNrHits.size();
  }
}

TrajSeedMatcher::TrajSeedMatcher(TrajectorySeedCollection const& seeds,
                                 math::XYZPoint const& vprim,
                                 TrajSeedMatcher::Configuration const& cfg,
                                 edm::EventSetup const& iSetup,
                                 MeasurementTrackerEvent const& measTkEvt)
    : seeds_{seeds},
      vprim_(vprim.x(), vprim.y(), vprim.z()),
      cfg_{cfg},
      magField_{iSetup.getData(cfg_.magFieldToken)},
      magFieldParam_{iSetup.getData(cfg_.paramMagFieldToken)},
      measTkEvt_{measTkEvt},
      navSchool_{iSetup.getData(cfg_.navSchoolToken)},
      detLayerGeom_{iSetup.getData(cfg_.detLayerGeomToken)},
      forwardPropagator_(alongMomentum, kElectronMass_, &magField_),
      backwardPropagator_(oppositeToMomentum, kElectronMass_, &magField_) {}

edm::ParameterSetDescription TrajSeedMatcher::makePSetDescription() {
  edm::ParameterSetDescription desc;
  desc.add<bool>("useRecoVertex", false);
  desc.add<bool>("enableHitSkipping", false);
  desc.add<bool>("requireExactMatchCount", true);
  desc.add<bool>("useParamMagFieldIfDefined", true);
  desc.add<edm::ESInputTag>("paramMagField", edm::ESInputTag{"", "ParabolicMf"});
  desc.add<edm::ESInputTag>("navSchool", edm::ESInputTag{"", "SimpleNavigationSchool"});
  desc.add<edm::ESInputTag>("detLayerGeom", edm::ESInputTag{"", "hltESPGlobalDetLayerGeometry"});
  desc.add<std::vector<int> >("minNrHitsValidLayerBins", {4});
  desc.add<std::vector<unsigned int> >("minNrHits", {2, 3});

  edm::ParameterSetDescription cutsDesc;
  auto cutDescCases = 1 >> (edm::ParameterDescription<double>("dPhiMax", 0.04, true) and
                            edm::ParameterDescription<double>("dRZMax", 0.09, true) and
                            edm::ParameterDescription<double>("dRZMaxLowEtThres", 20., true) and
                            edm::ParameterDescription<std::vector<double> >("dRZMaxLowEtEtaBins", {1., 1.5}, true) and
                            edm::ParameterDescription<std::vector<double> >("dRZMaxLowEt", {0.09, 0.15, 0.09}, true)) or
                      2 >> (edm::ParameterDescription<std::vector<double> >("dPhiMaxHighEt", {0.003}, true) and
                            edm::ParameterDescription<std::vector<double> >("dPhiMaxHighEtThres", {0.0}, true) and
                            edm::ParameterDescription<std::vector<double> >("dPhiMaxLowEtGrad", {0.0}, true) and
                            edm::ParameterDescription<std::vector<double> >("dRZMaxHighEt", {0.005}, true) and
                            edm::ParameterDescription<std::vector<double> >("dRZMaxHighEtThres", {30}, true) and
                            edm::ParameterDescription<std::vector<double> >("dRZMaxLowEtGrad", {-0.002}, true) and
                            edm::ParameterDescription<std::vector<double> >("etaBins", {}, true));
  cutsDesc.ifValue(edm::ParameterDescription<int>("version", 1, true), std::move(cutDescCases));

  edm::ParameterSet defaults;
  defaults.addParameter<double>("dPhiMax", 0.04);
  defaults.addParameter<double>("dRZMax", 0.09);
  defaults.addParameter<double>("dRZMaxLowEtThres", 0.09);
  defaults.addParameter<std::vector<double> >("dRZMaxLowEtEtaBins", std::vector<double>{1., 1.5});
  defaults.addParameter<std::vector<double> >("dRZMaxLowEt", std::vector<double>{0.09, 0.09, 0.09});
  defaults.addParameter<int>("version", 1);
  desc.addVPSet("matchingCuts", cutsDesc, std::vector<edm::ParameterSet>{defaults, defaults, defaults});
  return desc;
}

std::vector<TrajSeedMatcher::SeedWithInfo> TrajSeedMatcher::operator()(const GlobalPoint& candPos, const float energy) {
  clearCache();

  std::vector<SeedWithInfo> matchedSeeds;

  //these are super expensive functions
  TrajectoryStateOnSurface scTrajStateOnSurfNeg = makeTrajStateOnSurface(candPos, energy, -1);
  TrajectoryStateOnSurface scTrajStateOnSurfPos = makeTrajStateOnSurface(candPos, energy, 1);

  for (const auto& seed : seeds_) {
    std::vector<SCHitMatch> matchedHitsNeg = processSeed(seed, candPos, energy, scTrajStateOnSurfNeg);
    std::vector<SCHitMatch> matchedHitsPos = processSeed(seed, candPos, energy, scTrajStateOnSurfPos);

    int nrValidLayersPos = 0;
    int nrValidLayersNeg = 0;
    if (matchedHitsNeg.size() >= 2) {
      nrValidLayersNeg = getNrValidLayersAlongTraj(matchedHitsNeg[0], matchedHitsNeg[1], candPos, energy, -1);
    }
    if (matchedHitsPos.size() >= 2) {
      nrValidLayersPos = getNrValidLayersAlongTraj(matchedHitsPos[0], matchedHitsPos[1], candPos, energy, +1);
    }

    int nrValidLayers = std::max(nrValidLayersNeg, nrValidLayersPos);
    size_t nrHitsRequired = getNrHitsRequired(nrValidLayers);
    bool matchCountPasses;
    if (cfg_.requireExactMatchCount) {
      // If the input seed collection is not cross-cleaned, an exact match is necessary to
      // prevent redundant seeds.
      matchCountPasses = matchedHitsNeg.size() == nrHitsRequired || matchedHitsPos.size() == nrHitsRequired;
    } else {
      matchCountPasses = matchedHitsNeg.size() >= nrHitsRequired || matchedHitsPos.size() >= nrHitsRequired;
    }
    if (matchCountPasses) {
      matchedSeeds.push_back({seed, makeMatchInfoVector(matchedHitsPos, matchedHitsNeg), nrValidLayers});
    }
  }
  return matchedSeeds;
}

std::vector<TrajSeedMatcher::SCHitMatch> TrajSeedMatcher::processSeed(const TrajectorySeed& seed,
                                                                      const GlobalPoint& candPos,
                                                                      const float energy,
                                                                      const TrajectoryStateOnSurface& initialTrajState) {
  //next try passing these variables in once...
  const float candEta = candPos.eta();
  const float candEt = energy * std::sin(candPos.theta());
  const int charge = initialTrajState.charge();

  std::vector<SCHitMatch> matches;
  FreeTrajectoryState firstMatchFreeTraj;
  GlobalPoint prevHitPos;
  GlobalPoint vertex;
  const auto nCuts = cfg_.matchingCuts.size();
  for (size_t iHit = 0;
       matches.size() < nCuts && iHit < seed.nHits() && (cfg_.enableHitSkipping || iHit == matches.size());
       iHit++) {
    auto const& recHit = *(seed.recHits().begin() + iHit);

    if (!recHit.isValid()) {
      continue;
    }

    const bool doFirstMatch = matches.empty();

    auto const& trajState = doFirstMatch
                                ? getTrajStateFromVtx(recHit, initialTrajState, backwardPropagator_)
                                : getTrajStateFromPoint(recHit, firstMatchFreeTraj, prevHitPos, forwardPropagator_);
    if (!trajState.isValid()) {
      continue;
    }

    auto const& vtxForMatchObject = doFirstMatch ? vprim_ : vertex;
    auto match = makeSCHitMatch(vtxForMatchObject, trajState, recHit, candEt, candEta, candPos.phi(), charge, 1);

    if ((*cfg_.matchingCuts[matches.size()])(match)) {
      matches.push_back(match);
      if (doFirstMatch) {
        //now we can figure out the z vertex
        double zVertex = cfg_.useRecoVertex ? vprim_.z() : getZVtxFromExtrapolation(vprim_, match.hitPos, candPos);
        vertex = GlobalPoint(vprim_.x(), vprim_.y(), zVertex);
        firstMatchFreeTraj = ftsFromVertexToPoint(match.hitPos, vertex, energy, charge);
      }
      prevHitPos = match.hitPos;
    }
  }
  return matches;
}

// compute the z vertex from the candidate position and the found pixel hit
float TrajSeedMatcher::getZVtxFromExtrapolation(const GlobalPoint& primeVtxPos,
                                                const GlobalPoint& hitPos,
                                                const GlobalPoint& candPos) {
  auto sq = [](float x) { return x * x; };
  auto calRDiff = [sq](const GlobalPoint& p1, const GlobalPoint& p2) {
    return std::sqrt(sq(p2.x() - p1.x()) + sq(p2.y() - p1.y()));
  };
  const double r1Diff = calRDiff(primeVtxPos, hitPos);
  const double r2Diff = calRDiff(hitPos, candPos);
  return hitPos.z() - r1Diff * (candPos.z() - hitPos.z()) / r2Diff;
}

const TrajectoryStateOnSurface& TrajSeedMatcher::getTrajStateFromVtx(const TrackingRecHit& hit,
                                                                     const TrajectoryStateOnSurface& initialState,
                                                                     const PropagatorWithMaterial& propagator) {
  auto& trajStateFromVtxCache =
      initialState.charge() == 1 ? trajStateFromVtxPosChargeCache_ : trajStateFromVtxNegChargeCache_;

  auto key = hit.det()->gdetIndex();
  auto res = trajStateFromVtxCache.find(key);
  if (res != trajStateFromVtxCache.end())
    return res->second;
  else {  //doesnt exist, need to make it
    //FIXME: check for efficiency
    auto val = trajStateFromVtxCache.emplace(key, propagator.propagate(initialState, hit.det()->surface()));
    return val.first->second;
  }
}

const TrajectoryStateOnSurface& TrajSeedMatcher::getTrajStateFromPoint(const TrackingRecHit& hit,
                                                                       const FreeTrajectoryState& initialState,
                                                                       const GlobalPoint& point,
                                                                       const PropagatorWithMaterial& propagator) {
  auto& trajStateFromPointCache =
      initialState.charge() == 1 ? trajStateFromPointPosChargeCache_ : trajStateFromPointNegChargeCache_;

  auto key = std::make_pair(hit.det()->gdetIndex(), point);
  auto res = trajStateFromPointCache.find(key);
  if (res != trajStateFromPointCache.end())
    return res->second;
  else {  //doesnt exist, need to make it
    //FIXME: check for efficiency
    auto val = trajStateFromPointCache.emplace(key, propagator.propagate(initialState, hit.det()->surface()));
    return val.first->second;
  }
}

TrajectoryStateOnSurface TrajSeedMatcher::makeTrajStateOnSurface(const GlobalPoint& pos,
                                                                 const float energy,
                                                                 const int charge) const {
  auto freeTS = ftsFromVertexToPoint(pos, vprim_, energy, charge);
  return TrajectoryStateOnSurface(freeTS, *PerpendicularBoundPlaneBuilder{}(freeTS.position(), freeTS.momentum()));
}

void TrajSeedMatcher::clearCache() {
  trajStateFromVtxPosChargeCache_.clear();
  trajStateFromVtxNegChargeCache_.clear();
  trajStateFromPointPosChargeCache_.clear();
  trajStateFromPointNegChargeCache_.clear();
}

int TrajSeedMatcher::getNrValidLayersAlongTraj(
    const SCHitMatch& hit1, const SCHitMatch& hit2, const GlobalPoint& candPos, const float energy, const int charge) {
  double zVertex = cfg_.useRecoVertex ? vprim_.z() : getZVtxFromExtrapolation(vprim_, hit1.hitPos, candPos);
  GlobalPoint vertex(vprim_.x(), vprim_.y(), zVertex);

  auto firstMatchFreeTraj = ftsFromVertexToPoint(hit1.hitPos, vertex, energy, charge);
  auto const& secondHitTraj = getTrajStateFromPoint(hit2.hit, firstMatchFreeTraj, hit1.hitPos, forwardPropagator_);
  return getNrValidLayersAlongTraj(hit2.hit.geographicalId(), secondHitTraj);
}

int TrajSeedMatcher::getNrValidLayersAlongTraj(const DetId& hitId, const TrajectoryStateOnSurface& hitTrajState) const {
  const DetLayer* detLayer = detLayerGeom_.idToLayer(hitId);
  if (detLayer == nullptr)
    return 0;

  const FreeTrajectoryState& hitFreeState = *hitTrajState.freeState();
  auto const inLayers = navSchool_.compatibleLayers(*detLayer, hitFreeState, oppositeToMomentum);
  const auto outLayers = navSchool_.compatibleLayers(*detLayer, hitFreeState, alongMomentum);

  int nrValidLayers = 1;  //because our current hit is also valid and wont be included in the count otherwise
  for (auto layer : inLayers) {
    if (GeomDetEnumerators::isTrackerPixel(layer->subDetector())) {
      if (layerHasValidHits(*layer, hitTrajState, backwardPropagator_))
        nrValidLayers++;
    }
  }
  for (auto layer : outLayers) {
    if (GeomDetEnumerators::isTrackerPixel(layer->subDetector())) {
      if (layerHasValidHits(*layer, hitTrajState, forwardPropagator_))
        nrValidLayers++;
    }
  }
  return nrValidLayers;
}

bool TrajSeedMatcher::layerHasValidHits(const DetLayer& layer,
                                        const TrajectoryStateOnSurface& hitSurState,
                                        const Propagator& propToLayerFromState) const {
  //FIXME: do not hardcode with werid magic numbers stolen from ancient tracking code
  //its taken from https://cmssdt.cern.ch/dxr/CMSSW/source/RecoTracker/TrackProducer/interface/TrackProducerBase.icc#165
  //which inspires this code
  Chi2MeasurementEstimator estimator(30., -3.0, 0.5, 2.0, 0.5, 1.e12);  // same as defauts....

  const std::vector<GeometricSearchDet::DetWithState>& detWithState =
      layer.compatibleDets(hitSurState, propToLayerFromState, estimator);
  if (detWithState.empty())
    return false;
  else {
    DetId id = detWithState.front().first->geographicalId();
    MeasurementDetWithData measDet = measTkEvt_.idToDet(id);
    if (measDet.isActive())
      return true;
    else
      return false;
  }
}

size_t TrajSeedMatcher::getNrHitsRequired(const int nrValidLayers) const {
  for (size_t binNr = 0; binNr < cfg_.minNrHitsValidLayerBins.size(); binNr++) {
    if (nrValidLayers < cfg_.minNrHitsValidLayerBins[binNr])
      return cfg_.minNrHits[binNr];
  }
  return cfg_.minNrHits.back();
}

TrajSeedMatcher::MatchingCutsV1::MatchingCutsV1(const edm::ParameterSet& pset)
    : dPhiMax_(pset.getParameter<double>("dPhiMax")),
      dRZMax_(pset.getParameter<double>("dRZMax")),
      dRZMaxLowEtThres_(pset.getParameter<double>("dRZMaxLowEtThres")),
      dRZMaxLowEtEtaBins_(pset.getParameter<std::vector<double> >("dRZMaxLowEtEtaBins")),
      dRZMaxLowEt_(pset.getParameter<std::vector<double> >("dRZMaxLowEt")) {
  if (dRZMaxLowEtEtaBins_.size() + 1 != dRZMaxLowEt_.size()) {
    throw cms::Exception("InvalidConfig") << " dRZMaxLowEtEtaBins should be 1 less than dRZMaxLowEt when its "
                                          << dRZMaxLowEtEtaBins_.size() << " vs " << dRZMaxLowEt_.size();
  }
}

bool TrajSeedMatcher::MatchingCutsV1::operator()(const TrajSeedMatcher::SCHitMatch& scHitMatch) const {
  if (dPhiMax_ >= 0 && std::abs(scHitMatch.dPhi) > dPhiMax_)
    return false;

  const float dRZMax = getDRZCutValue(scHitMatch.et, scHitMatch.eta);
  if (dRZMax_ >= 0 && std::abs(scHitMatch.dRZ) > dRZMax)
    return false;

  return true;
}

float TrajSeedMatcher::MatchingCutsV1::getDRZCutValue(const float scEt, const float scEta) const {
  if (scEt >= dRZMaxLowEtThres_)
    return dRZMax_;
  else {
    const float absEta = std::abs(scEta);
    for (size_t etaNr = 0; etaNr < dRZMaxLowEtEtaBins_.size(); etaNr++) {
      if (absEta < dRZMaxLowEtEtaBins_[etaNr])
        return dRZMaxLowEt_[etaNr];
    }
    return dRZMaxLowEt_.back();
  }
}

TrajSeedMatcher::MatchingCutsV2::MatchingCutsV2(const edm::ParameterSet& pset)
    : dPhiHighEt_(pset.getParameter<std::vector<double> >("dPhiMaxHighEt")),
      dPhiHighEtThres_(pset.getParameter<std::vector<double> >("dPhiMaxHighEtThres")),
      dPhiLowEtGrad_(pset.getParameter<std::vector<double> >("dPhiMaxLowEtGrad")),
      dRZHighEt_(pset.getParameter<std::vector<double> >("dRZMaxHighEt")),
      dRZHighEtThres_(pset.getParameter<std::vector<double> >("dRZMaxHighEtThres")),
      dRZLowEtGrad_(pset.getParameter<std::vector<double> >("dRZMaxLowEtGrad")),
      etaBins_(pset.getParameter<std::vector<double> >("etaBins")) {
  auto binSizeCheck = [](size_t sizeEtaBins, const std::vector<double>& vec, const std::string& name) {
    if (vec.size() != sizeEtaBins + 1) {
      throw cms::Exception("InvalidConfig")
          << " when constructing TrajSeedMatcher::MatchingCutsV2 " << name << " has " << vec.size()
          << " bins, it should be equal to #bins of etaBins+1" << sizeEtaBins + 1;
    }
  };
  binSizeCheck(etaBins_.size(), dPhiHighEt_, "dPhiMaxHighEt");
  binSizeCheck(etaBins_.size(), dPhiHighEtThres_, "dPhiMaxHighEtThres");
  binSizeCheck(etaBins_.size(), dPhiLowEtGrad_, "dPhiMaxLowEtGrad");
  binSizeCheck(etaBins_.size(), dRZHighEt_, "dRZMaxHighEt");
  binSizeCheck(etaBins_.size(), dRZHighEtThres_, "dRZMaxHighEtThres");
  binSizeCheck(etaBins_.size(), dRZLowEtGrad_, "dRZMaxLowEtGrad");
}

bool TrajSeedMatcher::MatchingCutsV2::operator()(const TrajSeedMatcher::SCHitMatch& scHitMatch) const {
  size_t binNr = getBinNr(scHitMatch.eta);
  float dPhiMax = getCutValue(scHitMatch.et, dPhiHighEt_[binNr], dPhiHighEtThres_[binNr], dPhiLowEtGrad_[binNr]);
  if (dPhiMax >= 0 && std::abs(scHitMatch.dPhi) > dPhiMax)
    return false;
  float dRZMax = getCutValue(scHitMatch.et, dRZHighEt_[binNr], dRZHighEtThres_[binNr], dRZLowEtGrad_[binNr]);
  if (dRZMax >= 0 && std::abs(scHitMatch.dRZ) > dRZMax)
    return false;

  return true;
}

//eta bins is exactly 1 smaller than the vectors which will be accessed by this bin nr
size_t TrajSeedMatcher::MatchingCutsV2::getBinNr(float eta) const {
  const float absEta = std::abs(eta);
  for (size_t etaNr = 0; etaNr < etaBins_.size(); etaNr++) {
    if (absEta < etaBins_[etaNr])
      return etaNr;
  }
  return etaBins_.size();
}
