#include <string>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedHostCollection.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/ftsFromVertexToPoint.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"
#include "TrackingTools/RecoGeometry/interface/GlobalDetLayerGeometry.h"

constexpr float kElectronMass_ = 0.000511f;
constexpr int kDoubletRejectionValidLayerThreshold = 4;

class ElectronSeedConverter : public edm::global::EDProducer<> {
public:
  explicit ElectronSeedConverter(const edm::ParameterSet &);

  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const final;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

  inline FreeTrajectoryState ftsFromVertexToPoint(const GlobalPoint &point,
                                                  const GlobalPoint &vertex,
                                                  float energy,
                                                  int charge,
                                                  const MagneticField &magField) const {
    return trackingTools::ftsFromVertexToPoint(magField, point, vertex, energy, charge);
  }

private:
  // Tokens
  const edm::EDGetTokenT<TrajectorySeedCollection> initialSeedsToken_;
  const edm::EDGetTokenT<reco::ElectronSeedHostCollection> matchedEleSeedSoAToken_;
  const edm::EDGetTokenT<std::vector<reco::SuperClusterRef>> superClustersToken_;
  const edm::EDPutTokenT<reco::ElectronSeedCollection> putToken_;

  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const edm::EDGetTokenT<MeasurementTrackerEvent> measTkEvtToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<NavigationSchool, NavigationSchoolRecord> navSchoolToken_;
  const edm::ESGetToken<DetLayerGeometry, RecoGeometryRecord> detLayerGeomToken_;

  int getNrValidLayersAlongTraj(const TrajectorySeed &seed,
                                const reco::SuperClusterRef &scRef,
                                const reco::BeamSpot &beamSpot,
                                const PropagatorWithMaterial &forwardPropagator,
                                const PropagatorWithMaterial &backwardPropagator,
                                const NavigationSchool &navSchool,
                                const DetLayerGeometry &detLayerGeom,
                                const MeasurementTrackerEvent &measTkEvt,
                                const MagneticField &magField) const;

  bool layerHasValidHits(const DetLayer &layer,
                         const TrajectoryStateOnSurface &hitSurState,
                         const Propagator &propagator,
                         const MeasurementTrackerEvent &measTkEvt) const;

  static float getZVtxFromExtrapolation(const GlobalPoint &primeVtxPos,
                                        const GlobalPoint &hitPos,
                                        const GlobalPoint &candPos);
};

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

void ElectronSeedConverter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("initialSeeds", edm::InputTag{"hltElePixelSeedsCombined"});
  desc.add<edm::InputTag>("eleSeedsSoA", edm::InputTag{"ElectronNHitSeedAlpakaProducer"});
  desc.add<edm::InputTag>("superClusters", edm::InputTag{"hltEgammaSuperClustersToPixelMatch"});
  desc.add<edm::InputTag>("beamSpot", edm::InputTag{"hltOnlineBeamSpot"});
  desc.add<edm::InputTag>("measTkEvt", edm::InputTag{"hltMeasurementTrackerEvent"});
  desc.add<edm::ESInputTag>("navSchool", edm::ESInputTag{"", "SimpleNavigationSchool"});
  desc.add<edm::ESInputTag>("detLayerGeom", edm::ESInputTag{"", "hltESPGlobalDetLayerGeometry"});
  descriptions.add("electronSeedConverter", desc);
}

// -----------------------------------------------------------------------------

ElectronSeedConverter::ElectronSeedConverter(const edm::ParameterSet &pset)
    : initialSeedsToken_(consumes(pset.getParameter<edm::InputTag>("initialSeeds"))),
      matchedEleSeedSoAToken_(consumes(pset.getParameter<edm::InputTag>("eleSeedsSoA"))),
      superClustersToken_(consumes(pset.getParameter<edm::InputTag>("superClusters"))),
      putToken_(produces()),
      beamSpotToken_(consumes(pset.getParameter<edm::InputTag>("beamSpot"))),
      measTkEvtToken_(consumes(pset.getParameter<edm::InputTag>("measTkEvt"))),
      magFieldToken_(esConsumes()),
      navSchoolToken_(esConsumes(pset.getParameter<edm::ESInputTag>("navSchool"))),
      detLayerGeomToken_(esConsumes(pset.getParameter<edm::ESInputTag>("detLayerGeom"))) {}

// -----------------------------------------------------------------------------

void ElectronSeedConverter::produce(edm::StreamID, edm::Event &event, const edm::EventSetup &iSetup) const {
  auto const &view = event.get(matchedEleSeedSoAToken_).const_view();
  auto const &superClusterRefs = event.get(superClustersToken_);
  auto const &initialSeeds = event.get(initialSeedsToken_);

  auto const &beamSpot = event.get(beamSpotToken_);
  auto const &measTkEvt = event.get(measTkEvtToken_);

  auto const &magField = iSetup.getData(magFieldToken_);
  auto const &navSchool = iSetup.getData(navSchoolToken_);
  auto const &detLayerGeom = iSetup.getData(detLayerGeomToken_);

  PropagatorWithMaterial forwardPropagator(alongMomentum, kElectronMass_, &magField);
  PropagatorWithMaterial backwardPropagator(oppositeToMomentum, kElectronMass_, &magField);

  reco::ElectronSeedCollection eleSeeds;

  int nMatched = 0, nDoublets = 0, nDoubletsRejected = 0, nTriplets = 0;

  for (int i = 0; i < view.metadata().size(); ++i) {
    if (view[i].isMatched() == 0)
      continue;

    const auto currentView = view[i];
    const int matchedScID = currentView.matchedScID();
    const int seedID = currentView.id();

    if (matchedScID < 0 || (unsigned)matchedScID >= superClusterRefs.size() || seedID < 0 ||
        (unsigned)seedID >= initialSeeds.size()) {
      edm::LogWarning("ElectronSeedConverter") << "Index out of bounds: SC=" << matchedScID << " Seed=" << seedID;
      continue;
    }

    const reco::SuperClusterRef &scRef = superClusterRefs[matchedScID];
    const TrajectorySeed &matchedSeed = initialSeeds[seedID];
    ++nMatched;

    // --- Doublet rejection ---
    if (matchedSeed.nHits() == 2) {
      ++nDoublets;
      const int nrValidLayers = getNrValidLayersAlongTraj(matchedSeed,
                                                          scRef,
                                                          beamSpot,
                                                          forwardPropagator,
                                                          backwardPropagator,
                                                          navSchool,
                                                          detLayerGeom,
                                                          measTkEvt,
                                                          magField);
      if (nrValidLayers >= kDoubletRejectionValidLayerThreshold) {
        ++nDoubletsRejected;
        continue;
      }
    } else {
      ++nTriplets;
    }

    reco::ElectronSeed eleSeed(matchedSeed);
    eleSeed.setCaloCluster(reco::ElectronSeed::CaloClusterRef(scRef));

    const auto makePMVars = [currentView](const ushort nHit) -> reco::ElectronSeed::PMVars {
      reco::ElectronSeed::PMVars pmVars;
      pmVars.setDPhi(currentView.PMVars_dPhiPos()[nHit], currentView.PMVars_dPhiNeg()[nHit]);
      pmVars.setDRZ(currentView.PMVars_dRZPos()[nHit], currentView.PMVars_dRZNeg()[nHit]);
      return pmVars;
    };

    for (uint nHit = 0; nHit < matchedSeed.nHits(); nHit++) {
      eleSeed.addHitInfo(makePMVars(nHit));
    }

    eleSeeds.emplace_back(eleSeed);
  }

  // Print per-event doublet rejection summary
  edm::LogPrint("ElectronSeedConverter") << "[ElectronSeedConverter] matched=" << nMatched << "  doublets=" << nDoublets
                                         << "  doublets_rejected=" << nDoubletsRejected
                                         << "  (frac=" << (nDoublets > 0 ? 100.f * nDoubletsRejected / nDoublets : 0.f)
                                         << "%)"
                                         << "  triplets=" << nTriplets << "  accepted=" << (int)eleSeeds.size();

  event.emplace(putToken_, std::move(eleSeeds));
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

int ElectronSeedConverter::getNrValidLayersAlongTraj(const TrajectorySeed &seed,
                                                     const reco::SuperClusterRef &scRef,
                                                     const reco::BeamSpot &beamSpot,
                                                     const PropagatorWithMaterial &forwardPropagator,
                                                     const PropagatorWithMaterial &backwardPropagator,
                                                     const NavigationSchool &navSchool,
                                                     const DetLayerGeometry &detLayerGeom,
                                                     const MeasurementTrackerEvent &measTkEvt,
                                                     const MagneticField &magField) const {
  if (seed.nHits() < 2)
    return 0;

  auto it = seed.recHits().begin();
  const auto &recHit1 = *it;
  const auto &recHit2 = *(++it);

  const GlobalPoint vprim(beamSpot.x0(), beamSpot.y0(), beamSpot.z0());
  const GlobalPoint candPos(scRef->position().x(), scRef->position().y(), scRef->position().z());
  const float energy = scRef->energy();

  const double zVertex = getZVtxFromExtrapolation(vprim, recHit1.globalPosition(), candPos);
  const GlobalPoint vertex(vprim.x(), vprim.y(), zVertex);

  auto countLayersForCharge = [&](int charge) -> int {
    auto fts = ftsFromVertexToPoint(recHit1.globalPosition(), vertex, energy, charge, magField);

    const TrajectoryStateOnSurface secondHitTS = forwardPropagator.propagate(fts, recHit2.det()->surface());
    if (!secondHitTS.isValid())
      return 0;

    const DetLayer *detLayer = detLayerGeom.idToLayer(recHit2.geographicalId());
    if (!detLayer)
      return 0;

    const FreeTrajectoryState &hitFreeState = *secondHitTS.freeState();

    const auto inLayers = navSchool.compatibleLayers(*detLayer, hitFreeState, oppositeToMomentum);
    const auto outLayers = navSchool.compatibleLayers(*detLayer, hitFreeState, alongMomentum);

    int count = 1;  // count second hit layer

    for (const auto *layer : inLayers) {
      if (GeomDetEnumerators::isTrackerPixel(layer->subDetector())) {
        if (layerHasValidHits(*layer, secondHitTS, backwardPropagator, measTkEvt))
          ++count;
      }
    }

    for (const auto *layer : outLayers) {
      if (GeomDetEnumerators::isTrackerPixel(layer->subDetector())) {
        if (layerHasValidHits(*layer, secondHitTS, forwardPropagator, measTkEvt))
          ++count;
      }
    }

    return count;
  };

  return std::max(countLayersForCharge(+1), countLayersForCharge(-1));
}

bool ElectronSeedConverter::layerHasValidHits(const DetLayer &layer,
                                              const TrajectoryStateOnSurface &tsos,
                                              const Propagator &propagator,
                                              const MeasurementTrackerEvent &measTkEvt) const {
  Chi2MeasurementEstimator estimator(30., -3.0, 0.5, 2.0, 0.5, 1.e12);

  const auto &dets = layer.compatibleDets(tsos, propagator, estimator);
  if (dets.empty())
    return false;

  const DetId id = dets.front().first->geographicalId();
  auto measDet = measTkEvt.idToDet(id);
  return measDet.isActive() && !measDet.hasBadComponents(dets.front().second);
}

float ElectronSeedConverter::getZVtxFromExtrapolation(const GlobalPoint &primeVtxPos,
                                                      const GlobalPoint &hitPos,
                                                      const GlobalPoint &candPos) {
  auto sq = [](float x) { return x * x; };

  auto rdiff = [&](const GlobalPoint &a, const GlobalPoint &b) {
    return std::sqrt(sq(b.x() - a.x()) + sq(b.y() - a.y()));
  };

  const float r1 = rdiff(primeVtxPos, hitPos);
  const float r2 = rdiff(hitPos, candPos);

  return hitPos.z() - r1 * (candPos.z() - hitPos.z()) / r2;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronSeedConverter);
