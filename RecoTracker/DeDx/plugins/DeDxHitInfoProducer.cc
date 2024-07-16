// -*- C++ -*-
//
// Package:    DeDxHitInfoProducer
// Class:      DeDxHitInfoProducer
//
/**\class DeDxHitInfoProducer DeDxHitInfoProducer.cc RecoTracker/DeDx/plugins/DeDxHitInfoProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  loic Quertenmont (querten)
//         Created:  Mon Nov 21 14:09:02 CEST 2014
// Modifications: Tamas Almos Vami (2022)

// system include files
#include <memory>

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/DeDxHitInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "RecoTracker/DeDx/interface/GenericTruncatedAverageDeDxEstimator.h"
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShape.h"
#include "RecoTracker/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

using namespace reco;
using namespace std;
using namespace edm;

class DeDxHitInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit DeDxHitInfoProducer(const edm::ParameterSet&);
  ~DeDxHitInfoProducer() override;

private:
  void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  void makeCalibrationMap(const TrackerGeometry& tkGeom_);
  void processRec(reco::DeDxHitInfo&, const SiStripRecHit2D&, const LocalPoint&, const LocalVector&, const float&);
  void processHit(const TrackingRecHit* recHit,
                  const float trackMomentum,
                  const LocalVector& trackDirection,
                  reco::DeDxHitInfo& hitDeDxInfo,
                  std::vector<float>& hitMomentum,
                  const LocalPoint& hitLocalPos);

  // ----------member data ---------------------------
  const bool usePixel_;
  const bool useStrip_;
  const float theMeVperADCPixel_;
  const float theMeVperADCStrip_;

  const unsigned int minTrackHits_;
  const float minTrackPt_;
  const float minTrackPtPrescale_;
  const float maxTrackEta_;

  const std::string calibrationPath_;
  const bool useCalibration_;
  const bool doShapeTest_;
  const bool usePixelShape_;
  const bool storeMomentumAtHit_;

  const unsigned int lowPtTracksPrescalePass_, lowPtTracksPrescaleFail_;
  GenericTruncatedAverageDeDxEstimator lowPtTracksEstimator_;
  const float lowPtTracksDeDxThreshold_;
  const bool usePixelForPrescales_;

  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  const edm::EDGetTokenT<SiPixelClusterShapeCache> pixShapeCacheToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const edm::ESGetToken<ClusterShapeHitFilter, CkfComponentsRecord> clShapeToken_;
  edm::ESHandle<TrackerGeometry> tkGeom_;
  edm::ESHandle<ClusterShapeHitFilter> clShape_;
  edm::Handle<SiPixelClusterShapeCache> pixShapeCache_;

  std::vector<std::vector<float>> calibGains_;
  unsigned int offsetDU_;

  uint64_t xorshift128p(uint64_t state[2]) {
    uint64_t x = state[0];
    uint64_t const y = state[1];
    state[0] = y;
    x ^= x << 23;                              // a
    state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);  // b, c
    return state[1] + y;
  }
};

DeDxHitInfoProducer::DeDxHitInfoProducer(const edm::ParameterSet& iConfig)
    : usePixel_(iConfig.getParameter<bool>("usePixel")),
      useStrip_(iConfig.getParameter<bool>("useStrip")),
      theMeVperADCPixel_(iConfig.getParameter<double>("MeVperADCPixel")),
      theMeVperADCStrip_(iConfig.getParameter<double>("MeVperADCStrip")),
      minTrackHits_(iConfig.getParameter<unsigned>("minTrackHits")),
      minTrackPt_(iConfig.getParameter<double>("minTrackPt")),
      minTrackPtPrescale_(iConfig.getParameter<double>("minTrackPtPrescale")),
      maxTrackEta_(iConfig.getParameter<double>("maxTrackEta")),
      calibrationPath_(iConfig.getParameter<string>("calibrationPath")),
      useCalibration_(iConfig.getParameter<bool>("useCalibration")),
      doShapeTest_(iConfig.getParameter<bool>("shapeTest")),
      usePixelShape_(not iConfig.getParameter<edm::InputTag>("clusterShapeCache").label().empty()),
      storeMomentumAtHit_(iConfig.getParameter<bool>("storeMomentumAtHit")),
      lowPtTracksPrescalePass_(iConfig.getParameter<uint32_t>("lowPtTracksPrescalePass")),
      lowPtTracksPrescaleFail_(iConfig.getParameter<uint32_t>("lowPtTracksPrescaleFail")),
      lowPtTracksEstimator_(iConfig.getParameter<edm::ParameterSet>("lowPtTracksEstimatorParameters")),
      lowPtTracksDeDxThreshold_(iConfig.getParameter<double>("lowPtTracksDeDxThreshold")),
      usePixelForPrescales_(iConfig.getParameter<bool>("usePixelForPrescales")),
      tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
      pixShapeCacheToken_(consumes<SiPixelClusterShapeCache>(iConfig.getParameter<edm::InputTag>("clusterShapeCache"))),
      tkGeomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      clShapeToken_(
          esConsumes<ClusterShapeHitFilter, CkfComponentsRecord>(edm::ESInputTag("", "ClusterShapeHitFilter"))) {
  produces<reco::DeDxHitInfoCollection>();
  produces<reco::DeDxHitInfoAss>();
  produces<edm::ValueMap<int>>("prescale");
  if (storeMomentumAtHit_)
    produces<edm::ValueMap<std::vector<float>>>("momentumAtHit");

  if (!usePixel_ && !useStrip_)
    edm::LogError("DeDxHitsProducer") << "No Pixel Hits NOR Strip Hits will be saved.  Running this module is useless";
}

DeDxHitInfoProducer::~DeDxHitInfoProducer() = default;

// ------------ method called once each job just before starting event loop  ------------
void DeDxHitInfoProducer::beginRun(edm::Run const& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = iSetup.getHandle(tkGeomToken_);
  if (useCalibration_ && calibGains_.empty()) {
    offsetDU_ = tkGeom_->offsetDU(GeomDetEnumerators::PixelBarrel);  //index start at the first pixel

    deDxTools::makeCalibrationMap(calibrationPath_, *tkGeom_, calibGains_, offsetDU_);
  }
}

void DeDxHitInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByToken(tracksToken_, trackCollectionHandle);
  const TrackCollection& trackCollection(*trackCollectionHandle.product());

  clShape_ = iSetup.getHandle(clShapeToken_);
  if (usePixelShape_)
    pixShapeCache_ = iEvent.getHandle(pixShapeCacheToken_);

  // creates the output collection
  auto resultdedxHitColl = std::make_unique<reco::DeDxHitInfoCollection>();

  std::vector<int> indices;
  std::vector<int> prescales;
  std::vector<std::vector<float>> hitMomenta;
  uint64_t state[2] = {iEvent.id().event(), iEvent.id().luminosityBlock()};
  for (unsigned int j = 0; j < trackCollection.size(); j++) {
    const reco::Track& track = trackCollection[j];

    //track selection
    bool passPt = (track.pt() >= minTrackPt_), passLowDeDx = false, passHighDeDx = false, pass = passPt;
    if (!pass && (track.pt() >= minTrackPtPrescale_)) {
      if (lowPtTracksPrescalePass_ > 0) {
        passHighDeDx = ((xorshift128p(state) % lowPtTracksPrescalePass_) == 0);
      }
      if (lowPtTracksPrescaleFail_ > 0) {
        passLowDeDx = ((xorshift128p(state) % lowPtTracksPrescaleFail_) == 0);
      }
      pass = passHighDeDx || passLowDeDx;
    }
    if (!pass || std::abs(track.eta()) > maxTrackEta_ || track.numberOfValidHits() < minTrackHits_) {
      indices.push_back(-1);
      continue;
    }

    reco::DeDxHitInfo hitDeDxInfo;
    std::vector<float> hitMomentum;
    auto const& trajParams = track.extra()->trajParams();
    auto hb = track.recHitsBegin();
    for (unsigned int h = 0; h < track.recHitsSize(); h++) {
      auto recHit = *(hb + h);
      if (!trackerHitRTTI::isFromDet(*recHit))
        continue;

      const auto& traj = trajParams[h];
      processHit(recHit, traj.momentum().mag(), traj.direction(), hitDeDxInfo, hitMomentum, traj.position());
    }
    assert(!storeMomentumAtHit_ || hitMomentum.size() == hitDeDxInfo.size());

    if (!passPt) {
      std::vector<DeDxHit> hits;
      hits.reserve(hitDeDxInfo.size());
      for (unsigned int i = 0, n = hitDeDxInfo.size(); i < n; ++i) {
        if (hitDeDxInfo.detId(i).subdetId() <= 2 && usePixelForPrescales_) {
          hits.push_back(DeDxHit(hitDeDxInfo.charge(i) / hitDeDxInfo.pathlength(i) * theMeVperADCPixel_, 0, 0, 0));
        } else if (hitDeDxInfo.detId(i).subdetId() > 2) {
          if (doShapeTest_ && !deDxTools::shapeSelection(*hitDeDxInfo.stripCluster(i)))
            continue;
          hits.push_back(DeDxHit(hitDeDxInfo.charge(i) / hitDeDxInfo.pathlength(i) * theMeVperADCStrip_, 0, 0, 0));
        }
      }

      // In case we have a pixel only track, but usePixelForPrescales_ is false
      if (hits.empty()) {
        indices.push_back(-1);
        continue;
      }
      std::sort(hits.begin(), hits.end(), std::less<DeDxHit>());
      if (lowPtTracksEstimator_.dedx(hits).first < lowPtTracksDeDxThreshold_) {
        if (passLowDeDx) {
          prescales.push_back(lowPtTracksPrescaleFail_);
        } else {
          indices.push_back(-1);
          continue;
        }
      } else {
        if (passHighDeDx) {
          prescales.push_back(lowPtTracksPrescalePass_);
        } else {
          indices.push_back(-1);
          continue;
        }
      }
    } else {
      prescales.push_back(1);
    }
    indices.push_back(resultdedxHitColl->size());
    resultdedxHitColl->push_back(hitDeDxInfo);
    if (storeMomentumAtHit_)
      hitMomenta.push_back(hitMomentum);
  }
  ///////////////////////////////////////

  edm::OrphanHandle<reco::DeDxHitInfoCollection> dedxHitCollHandle = iEvent.put(std::move(resultdedxHitColl));

  //create map passing the handle to the matched collection
  auto dedxMatch = std::make_unique<reco::DeDxHitInfoAss>(dedxHitCollHandle);
  reco::DeDxHitInfoAss::Filler filler(*dedxMatch);
  filler.insert(trackCollectionHandle, indices.begin(), indices.end());
  filler.fill();
  iEvent.put(std::move(dedxMatch));

  auto dedxPrescale = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler pfiller(*dedxPrescale);
  pfiller.insert(dedxHitCollHandle, prescales.begin(), prescales.end());
  pfiller.fill();
  iEvent.put(std::move(dedxPrescale), "prescale");

  if (storeMomentumAtHit_) {
    auto dedxMomenta = std::make_unique<edm::ValueMap<std::vector<float>>>();
    edm::ValueMap<std::vector<float>>::Filler mfiller(*dedxMomenta);
    mfiller.insert(dedxHitCollHandle, hitMomenta.begin(), hitMomenta.end());
    mfiller.fill();
    iEvent.put(std::move(dedxMomenta), "momentumAtHit");
  }
}

void DeDxHitInfoProducer::processRec(reco::DeDxHitInfo& hitDeDxInfo,
                                     const SiStripRecHit2D& recHit,
                                     const LocalPoint& lpos,
                                     const LocalVector& ldir,
                                     const float& cos) {
  uint8_t type(0);
  int meas;
  float pred;
  const auto& usable = clShape_->getSizes(recHit, {}, ldir, meas, pred);
  if (usable && meas <= int(std::abs(pred)) + 4)
    type |= (1 << reco::DeDxHitInfo::Complete);
  if (clShape_->isCompatible(recHit, ldir))
    type |= (1 << reco::DeDxHitInfo::Compatible);
  if (usable)
    type |= (1 << reco::DeDxHitInfo::Calibration);

  int NSaturating(0);
  const auto& detId = recHit.geographicalId();
  const auto* detUnit = recHit.detUnit();
  if (detUnit == nullptr)
    detUnit = tkGeom_->idToDet(detId);
  const auto pathLen = detUnit->surface().bounds().thickness() / cos;
  float chargeAbs = deDxTools::getCharge(&(recHit.stripCluster()), NSaturating, *detUnit, calibGains_, offsetDU_);
  hitDeDxInfo.addHit(chargeAbs, pathLen, detId, lpos, type, recHit.stripCluster());
}

void DeDxHitInfoProducer::processHit(const TrackingRecHit* recHit,
                                     const float trackMomentum,
                                     const LocalVector& trackDirection,
                                     reco::DeDxHitInfo& hitDeDxInfo,
                                     std::vector<float>& hitMomentum,
                                     const LocalPoint& hitLocalPos) {
  auto const& thit = static_cast<BaseTrackerRecHit const&>(*recHit);
  if (!thit.isValid())
    return;

  //make sure cosine is not 0
  float cosine = trackDirection.z() / trackDirection.mag();
  float cosineAbs = std::max(0.00000001f, std::abs(cosine));

  auto const& clus = thit.firstClusterRef();
  if (!clus.isValid())
    return;

  const auto* detUnit = recHit->detUnit();
  if (detUnit == nullptr) {
    detUnit = tkGeom_->idToDet(thit.geographicalId());
  }
  float pathLen = detUnit->surface().bounds().thickness() / cosineAbs;

  if (clus.isPixel()) {
    if (!usePixel_)
      return;

    uint8_t type(0);
    const auto& pixelDet = *dynamic_cast<const PixelGeomDetUnit*>(detUnit);
    const auto& pixelRecHit = *dynamic_cast<const SiPixelRecHit*>(recHit);
    ClusterData data;
    ClusterShape().determineShape(pixelDet, clus.pixelCluster(), data);
    if (data.isComplete)
      type |= (1 << reco::DeDxHitInfo::Complete);
    if (usePixelShape_ && clShape_->isCompatible(pixelRecHit, trackDirection, *pixShapeCache_))
      type |= (1 << reco::DeDxHitInfo::Compatible);
    if (data.isComplete && data.isStraight && data.hasBigPixelsOnlyInside)
      type |= (1 << reco::DeDxHitInfo::Calibration);

    float chargeAbs = clus.pixelCluster().charge();
    hitDeDxInfo.addHit(chargeAbs, pathLen, thit.geographicalId(), hitLocalPos, type, clus.pixelCluster());
  } else if (clus.isStrip() && !thit.isMatched()) {
    if (!useStrip_)
      return;

    processRec(hitDeDxInfo, {thit.geographicalId(), clus}, hitLocalPos, trackDirection, cosineAbs);
  } else if (clus.isStrip() && thit.isMatched()) {
    if (!useStrip_)
      return;
    const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D*>(recHit);
    if (!matchedHit)
      return;
    if (storeMomentumAtHit_)
      hitMomentum.push_back(trackMomentum);

    processRec(hitDeDxInfo, matchedHit->monoHit(), hitLocalPos, trackDirection, cosineAbs);
    processRec(hitDeDxInfo, matchedHit->stereoHit(), hitLocalPos, trackDirection, cosineAbs);
  }
  if (storeMomentumAtHit_ && (clus.isPixel() || clus.isStrip()))
    hitMomentum.push_back(trackMomentum);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeDxHitInfoProducer);
