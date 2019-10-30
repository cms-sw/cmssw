// -*- C++ -*-
//
// Package:    FastTrackDeDxProducer
// Class:      FastTrackDeDxProducer
//
/**\class FastTrackDeDxProducer FastTrackDeDxProducer.cc RecoTracker/FastTrackDeDxProducer/src/FastTrackDeDxProducer.cc

   Description: <one line class summary>

   Implementation:
   <Notes on implementation>
*/
// Original author:  Sam Bein
//         Created:  Wednesday Dec 26 14:17:19 CEST 2018
// Author of derivative code:  andrea
//         Created:  Thu May 31 14:09:02 CEST 2007
// Code Updates:  loic Quertenmont (querten)
//         Created:  Thu May 10 14:09:02 CEST 2008
//
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/GenericAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/TruncatedAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/MedianDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/UnbinnedFitDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/ProductDeDxDiscriminator.h"
#include "RecoTracker/DeDx/interface/SmirnovDeDxDiscriminator.h"
#include "RecoTracker/DeDx/interface/ASmirnovDeDxDiscriminator.h"
#include "RecoTracker/DeDx/interface/BTagLikeDeDxDiscriminator.h"

#include "RecoTracker/DeDx/interface/DeDxTools.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"

//
// class declaration
//

class FastTrackDeDxProducer : public edm::stream::EDProducer<> {
public:
  explicit FastTrackDeDxProducer(const edm::ParameterSet&);
  ~FastTrackDeDxProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  void makeCalibrationMap(const TrackerGeometry& tkGeom);
  void processHit(const FastTrackerRecHit& recHit,
                  float trackMomentum,
                  float& cosine,
                  reco::DeDxHitCollection& dedxHits,
                  int& NClusterSaturating);

  // ----------member data ---------------------------
  //BaseDeDxEstimator*               m_estimator;

  std::unique_ptr<BaseDeDxEstimator> m_estimator;

  edm::EDGetTokenT<reco::TrackCollection> m_tracksTag;

  float meVperADCPixel;
  float meVperADCStrip;

  unsigned int MaxNrStrips;

  std::string m_calibrationPath;

  std::vector<std::vector<float>> calibGains;
  unsigned int m_off;

  edm::EDGetTokenT<edm::PSimHitContainer> simHitsToken;
  edm::EDGetTokenT<FastTrackerRecHitRefCollection> simHit2RecHitMapToken;

  bool usePixel;
  bool useStrip;
  bool useCalibration;
  bool shapetest;
  bool convertFromGeV2MeV;
  bool nothick;
};

using namespace reco;
using namespace std;
using namespace edm;

void FastTrackDeDxProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<string>("estimator", "generic");
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<bool>("UsePixel", false);
  desc.add<bool>("UseStrip", true);
  desc.add<double>("MeVperADCStrip", 3.61e-06 * 265);
  desc.add<double>("MeVperADCPixel", 3.61e-06);
  desc.add<bool>("ShapeTest", true);
  desc.add<bool>("UseCalibration", false);
  desc.add<string>("calibrationPath", "");
  desc.add<string>("Reccord", "SiStripDeDxMip_3D_Rcd");
  desc.add<string>("ProbabilityMode", "Accumulation");
  desc.add<double>("fraction", 0.4);
  desc.add<double>("exponent", -2.0);
  desc.add<bool>("convertFromGeV2MeV", true);
  desc.add<bool>("nothick", false);
  desc.add<edm::InputTag>("simHits");
  desc.add<edm::InputTag>("simHit2RecHitMap");
  descriptions.add("FastTrackDeDxProducer", desc);
}

FastTrackDeDxProducer::FastTrackDeDxProducer(const edm::ParameterSet& iConfig)
    : simHitsToken(consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("simHits"))),
      simHit2RecHitMapToken(
          consumes<FastTrackerRecHitRefCollection>(iConfig.getParameter<edm::InputTag>("simHit2RecHitMap"))) {
  produces<ValueMap<DeDxData>>();

  string estimatorName = iConfig.getParameter<string>("estimator");
  if (estimatorName == "median")
    m_estimator = std::unique_ptr<BaseDeDxEstimator>(new MedianDeDxEstimator(iConfig));
  else if (estimatorName == "generic")
    m_estimator = std::unique_ptr<BaseDeDxEstimator>(new GenericAverageDeDxEstimator(iConfig));
  else if (estimatorName == "truncated")
    m_estimator = std::unique_ptr<BaseDeDxEstimator>(new TruncatedAverageDeDxEstimator(iConfig));
  //else if(estimatorName == "unbinnedFit")         m_estimator = std::unique_ptr<BaseDeDxEstimator> (new UnbinnedFitDeDxEstimator(iConfig));//estimator used in FullSimVersion
  else if (estimatorName == "productDiscrim")
    m_estimator = std::unique_ptr<BaseDeDxEstimator>(new ProductDeDxDiscriminator(iConfig));
  else if (estimatorName == "btagDiscrim")
    m_estimator = std::unique_ptr<BaseDeDxEstimator>(new BTagLikeDeDxDiscriminator(iConfig));
  else if (estimatorName == "smirnovDiscrim")
    m_estimator = std::unique_ptr<BaseDeDxEstimator>(new SmirnovDeDxDiscriminator(iConfig));
  else if (estimatorName == "asmirnovDiscrim")
    m_estimator = std::unique_ptr<BaseDeDxEstimator>(new ASmirnovDeDxDiscriminator(iConfig));
  else
    throw cms::Exception("fastsim::SimplifiedGeometry::FastTrackDeDxProducer.cc") << " estimator name does not exist";

  //Commented for now, might be used in the future
  //   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  255);

  m_tracksTag = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));

  usePixel = iConfig.getParameter<bool>("UsePixel");
  useStrip = iConfig.getParameter<bool>("UseStrip");
  meVperADCPixel = iConfig.getParameter<double>("MeVperADCPixel");
  meVperADCStrip = iConfig.getParameter<double>("MeVperADCStrip");

  shapetest = iConfig.getParameter<bool>("ShapeTest");
  useCalibration = iConfig.getParameter<bool>("UseCalibration");
  m_calibrationPath = iConfig.getParameter<string>("calibrationPath");

  convertFromGeV2MeV = iConfig.getParameter<bool>("convertFromGeV2MeV");
  nothick = iConfig.getParameter<bool>("nothick");

  if (!usePixel && !useStrip)
    throw cms::Exception("fastsim::SimplifiedGeometry::FastTrackDeDxProducer.cc")
        << " neither pixel hits nor strips hits will be used to compute de/dx";
}

// ------------ method called once each job just before starting event loop  ------------
void FastTrackDeDxProducer::beginRun(edm::Run const& run, const edm::EventSetup& iSetup) {
  if (useCalibration && calibGains.empty()) {
    edm::ESHandle<TrackerGeometry> tkGeom;
    iSetup.get<TrackerDigiGeometryRecord>().get(tkGeom);
    m_off = tkGeom->offsetDU(GeomDetEnumerators::PixelBarrel);  //index start at the first pixel

    DeDxTools::makeCalibrationMap(m_calibrationPath, *tkGeom, calibGains, m_off);
  }

  m_estimator->beginRun(run, iSetup);
}

void FastTrackDeDxProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto trackDeDxEstimateAssociation = std::make_unique<ValueMap<DeDxData>>();
  ValueMap<DeDxData>::Filler filler(*trackDeDxEstimateAssociation);

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByToken(m_tracksTag, trackCollectionHandle);
  const auto& trackCollection = *trackCollectionHandle;
  std::vector<DeDxData> dedxEstimate(trackCollection.size());

  for (unsigned int j = 0; j < trackCollection.size(); j++) {
    const reco::TrackRef track = reco::TrackRef(trackCollectionHandle.product(), j);

    int NClusterSaturating = 0;
    DeDxHitCollection dedxHits;

    auto const& trajParams = track->extra()->trajParams();
    assert(trajParams.size() == track->recHitsSize());

    auto hb = track->recHitsBegin();
    dedxHits.reserve(track->recHitsSize() / 2);

    for (unsigned int h = 0; h < track->recHitsSize(); h++) {
      const FastTrackerRecHit& recHit = static_cast<const FastTrackerRecHit&>(*(*(hb + h)));
      if (!recHit.isValid())
        continue;  //FastTrackerRecHit recHit = *(hb+h);
      auto trackDirection = trajParams[h].direction();
      float cosine = trackDirection.z() / trackDirection.mag();
      processHit(recHit, track->p(), cosine, dedxHits, NClusterSaturating);
    }

    sort(dedxHits.begin(), dedxHits.end(), less<DeDxHit>());
    std::pair<float, float> val_and_error = m_estimator->dedx(dedxHits);
    //WARNING: Since the dEdX Error is not properly computed for the moment
    //It was decided to store the number of saturating cluster in that dataformat
    val_and_error.second = NClusterSaturating;
    dedxEstimate[j] = DeDxData(val_and_error.first, val_and_error.second, dedxHits.size());
  }

  filler.insert(trackCollectionHandle, dedxEstimate.begin(), dedxEstimate.end());
  // fill the association map and put it into the event
  filler.fill();
  iEvent.put(std::move(trackDeDxEstimateAssociation));
}

void FastTrackDeDxProducer::processHit(const FastTrackerRecHit& recHit,
                                       float trackMomentum,
                                       float& cosine,
                                       reco::DeDxHitCollection& dedxHits,
                                       int& NClusterSaturating) {
  if (!recHit.isValid())
    return;

  auto const& thit = static_cast<BaseTrackerRecHit const&>(recHit);
  if (!thit.isValid())
    return;
  if (!thit.hasPositionAndError())
    return;

  if (recHit.isPixel()) {
    if (!usePixel)
      return;

    auto& detUnit = *(recHit.detUnit());
    float pathLen = detUnit.surface().bounds().thickness() / fabs(cosine);
    if (nothick)
      pathLen = 1.0;
    float charge = recHit.energyLoss() / pathLen;
    if (convertFromGeV2MeV)
      charge *= 1000;
    dedxHits.push_back(DeDxHit(charge, trackMomentum, pathLen, recHit.geographicalId()));
  } else if (!recHit.isPixel()) {  // && !recHit.isMatched()){//check what recHit.isMatched is doing
    if (!useStrip)
      return;
    auto& detUnit = *(recHit.detUnit());
    float pathLen = detUnit.surface().bounds().thickness() / fabs(cosine);
    if (nothick)
      pathLen = 1.0;
    float dedxOfRecHit = recHit.energyLoss() / pathLen;
    if (convertFromGeV2MeV)
      dedxOfRecHit *= 1000;
    if (!shapetest) {
      dedxHits.push_back(DeDxHit(dedxOfRecHit, trackMomentum, pathLen, recHit.geographicalId()));
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(FastTrackDeDxProducer);
