#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoHI/HiTracking/interface/HIPixelTrackFilter.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

class HIPixelTrackFilterProducer : public edm::global::EDProducer<> {
public:
  explicit HIPixelTrackFilterProducer(const edm::ParameterSet& iConfig);
  ~HIPixelTrackFilterProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<SiPixelClusterShapeCache> theClusterShapeCacheToken;
  edm::EDGetTokenT<reco::VertexCollection> theVertexCollectionToken;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> theTrackerToken;
  edm::ESGetToken<ClusterShapeHitFilter, CkfComponentsRecord> theShapeToken;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> theTopoToken;
  double theTIPMax, theNSigmaTipMaxTolerance;
  double theLIPMax, theNSigmaLipMaxTolerance;
  double theChi2Max, thePtMin, thePtMax;
  bool useClusterShape;
};

HIPixelTrackFilterProducer::HIPixelTrackFilterProducer(const edm::ParameterSet& iConfig)
    : theClusterShapeCacheToken(
          consumes<SiPixelClusterShapeCache>(iConfig.getParameter<edm::InputTag>("clusterShapeCacheSrc"))),
      theVertexCollectionToken(
          consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("VertexCollection"))),
      theTrackerToken(esConsumes()),
      theShapeToken(esConsumes(edm::ESInputTag("", "ClusterShapeHitFilter"))),
      theTopoToken(esConsumes()),
      theTIPMax(iConfig.getParameter<double>("tipMax")),
      theNSigmaTipMaxTolerance(iConfig.getParameter<double>("nSigmaTipMaxTolerance")),
      theLIPMax(iConfig.getParameter<double>("lipMax")),
      theNSigmaLipMaxTolerance(iConfig.getParameter<double>("nSigmaLipMaxTolerance")),
      theChi2Max(iConfig.getParameter<double>("chi2")),
      thePtMin(iConfig.getParameter<double>("ptMin")),
      thePtMax(iConfig.getParameter<double>("ptMax")),
      useClusterShape(iConfig.getParameter<bool>("useClusterShape")) {
  produces<PixelTrackFilter>();
}

void HIPixelTrackFilterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("clusterShapeCacheSrc", edm::InputTag("siPixelClusterShapeCache"));
  desc.add<edm::InputTag>("VertexCollection", edm::InputTag("hiSelectedPixelVertex"));
  desc.add<double>("ptMin", 1.5);
  desc.add<double>("ptMax", 999999.);
  desc.add<double>("tipMax", 0);
  desc.add<double>("nSigmaTipMaxTolerance", 6.0);
  desc.add<double>("lipMax", 0.3);
  desc.add<double>("nSigmaLipMaxTolerance", 0);
  desc.add<double>("chi2", 1000);
  desc.add<bool>("useClusterShape", false);

  descriptions.add("hiPixelTrackFilter", desc);
}

HIPixelTrackFilterProducer::~HIPixelTrackFilterProducer() {}

void HIPixelTrackFilterProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<SiPixelClusterShapeCache> cache;
  iEvent.getByToken(theClusterShapeCacheToken, cache);

  edm::Handle<reco::VertexCollection> vc;
  iEvent.getByToken(theVertexCollectionToken, vc);
  const reco::VertexCollection* vertices = vc.product();

  if (!vertices->empty()) {
    edm::LogInfo("HeavyIonVertexing") << "[HIPixelTrackFilterProducer] Pixel track selection based on best vertex"
                                      << "\n   vz = " << vertices->begin()->z()
                                      << "\n   vz sigma = " << vertices->begin()->zError();
  } else {
    edm::EDConsumerBase::Labels labels;
    labelsForToken(theVertexCollectionToken, labels);

    edm::LogError("HeavyIonVertexing")  // this can be made a warning when operator() is fixed
        << "No vertex found in collection '" << labels.module << "'";
  }

  auto impl = std::make_unique<HIPixelTrackFilter>(cache.product(),
                                                   thePtMin,
                                                   thePtMax,
                                                   &iSetup.getData(theTrackerToken),
                                                   &iSetup.getData(theShapeToken),
                                                   &iSetup.getData(theTopoToken),
                                                   vertices,
                                                   theTIPMax,
                                                   theNSigmaTipMaxTolerance,
                                                   theLIPMax,
                                                   theNSigmaLipMaxTolerance,
                                                   theChi2Max,
                                                   useClusterShape);
  auto prod = std::make_unique<PixelTrackFilter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(HIPixelTrackFilterProducer);
