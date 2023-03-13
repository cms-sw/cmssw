#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoHI/HiTracking/interface/HIProtoTrackFilter.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/Common/interface/DetSetAlgorithm.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class HIProtoTrackFilterProducer : public edm::global::EDProducer<> {
public:
  explicit HIProtoTrackFilterProducer(const edm::ParameterSet& iConfig);
  ~HIProtoTrackFilterProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<reco::BeamSpot> theBeamSpotToken;
  edm::EDGetTokenT<SiPixelRecHitCollection> theSiPixelRecHitsToken;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> theTtopoToken;
  double theTIPMax;
  double theChi2Max, thePtMin;
  bool doVariablePtMin;
};

HIProtoTrackFilterProducer::HIProtoTrackFilterProducer(const edm::ParameterSet& iConfig)
    : theBeamSpotToken(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      theSiPixelRecHitsToken(consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("siPixelRecHits"))),
      theTtopoToken(esConsumes()),
      theTIPMax(iConfig.getParameter<double>("tipMax")),
      theChi2Max(iConfig.getParameter<double>("chi2")),
      thePtMin(iConfig.getParameter<double>("ptMin")),
      doVariablePtMin(iConfig.getParameter<bool>("doVariablePtMin")) {
  produces<PixelTrackFilter>();
}

void HIProtoTrackFilterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("siPixelRecHits", edm::InputTag("siPixelRecHits"));
  desc.add<double>("ptMin", 1.0);
  desc.add<double>("tipMax", 1.0);
  desc.add<double>("chi2", 1000);
  desc.add<bool>("doVariablePtMin", true);

  descriptions.add("hiProtoTrackFilter", desc);
}

HIProtoTrackFilterProducer::~HIProtoTrackFilterProducer() {}

void HIProtoTrackFilterProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Get the beam spot
  edm::Handle<reco::BeamSpot> bsHandle;
  iEvent.getByToken(theBeamSpotToken, bsHandle);
  const reco::BeamSpot* beamSpot = bsHandle.product();

  if (beamSpot) {
    edm::LogInfo("HeavyIonVertexing") << "[HIProtoTrackFilterProducer] Proto track selection based on beamspot"
                                      << "\n   (x,y,z) = (" << beamSpot->x0() << "," << beamSpot->y0() << ","
                                      << beamSpot->z0() << ")";
  } else {
    edm::EDConsumerBase::Labels labels;
    labelsForToken(theBeamSpotToken, labels);
    edm::LogError("HeavyIonVertexing")  // this can be made a warning when operator() is fixed
        << "No beamspot found with tag '" << labels.module << "'";
  }

  // Estimate multiplicity
  edm::Handle<SiPixelRecHitCollection> recHitColl;
  iEvent.getByToken(theSiPixelRecHitsToken, recHitColl);

  auto const& ttopo = iSetup.getData(theTtopoToken);

  std::vector<const TrackingRecHit*> theChosenHits;
  edmNew::copyDetSetRange(*recHitColl, theChosenHits, ttopo.pxbDetIdLayerComparator(1));
  float estMult = theChosenHits.size();

  double variablePtMin = thePtMin;
  if (doVariablePtMin) {
    // parameterize ptMin such that a roughly constant number of selected prototracks passed are to vertexing
    float varPtCutoff = 1500;  //cutoff for variable ptMin
    if (estMult < varPtCutoff) {
      variablePtMin = 0.075;
      if (estMult > 0)
        variablePtMin = (13. - (varPtCutoff / estMult)) / 12.;
      if (variablePtMin < 0.075)
        variablePtMin = 0.075;  // don't lower the cut past 75 MeV
    }
    LogTrace("heavyIonHLTVertexing") << "   [HIProtoTrackFilterProducer: variablePtMin: " << variablePtMin << "]";
  }

  auto impl = std::make_unique<HIProtoTrackFilter>(beamSpot, theTIPMax, theChi2Max, variablePtMin);
  auto prod = std::make_unique<PixelTrackFilter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(HIProtoTrackFilterProducer);
