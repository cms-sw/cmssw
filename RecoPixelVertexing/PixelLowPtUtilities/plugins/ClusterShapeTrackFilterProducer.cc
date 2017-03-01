#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrackFilter.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"

class ClusterShapeTrackFilterProducer: public edm::global::EDProducer<> {
public:
  explicit ClusterShapeTrackFilterProducer(const edm::ParameterSet& iConfig);
  ~ClusterShapeTrackFilterProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<SiPixelClusterShapeCache> clusterShapeCacheToken_;
  const double ptMin_;
  const double ptMax_;
};

ClusterShapeTrackFilterProducer::ClusterShapeTrackFilterProducer(const edm::ParameterSet& iConfig):
  clusterShapeCacheToken_(consumes<SiPixelClusterShapeCache>(iConfig.getParameter<edm::InputTag>("clusterShapeCacheSrc"))),
  ptMin_(iConfig.getParameter<double>("ptMin")),
  ptMax_(iConfig.getParameter<double>("ptMax"))
{
  produces<PixelTrackFilter>();
}

void ClusterShapeTrackFilterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("clusterShapeCacheSrc", edm::InputTag("siPixelClusterShapeCache"));
  desc.add<double>("ptMin", 0);
  desc.add<double>("ptMax", 999999.);

  descriptions.add("clusterShapeTrackFilter", desc);
}

ClusterShapeTrackFilterProducer::~ClusterShapeTrackFilterProducer() {}

void ClusterShapeTrackFilterProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<SiPixelClusterShapeCache> cache;
  iEvent.getByToken(clusterShapeCacheToken_, cache);

  auto impl = std::make_unique<ClusterShapeTrackFilter>(cache.product(), ptMin_, ptMax_, iSetup);
  auto prod = std::make_unique<PixelTrackFilter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(ClusterShapeTrackFilterProducer);
