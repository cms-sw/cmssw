#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShape.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"

#include <cassert>

class SiPixelClusterShapeCacheProducer: public edm::stream::EDProducer<> {
public:
  explicit SiPixelClusterShapeCacheProducer(const edm::ParameterSet& iConfig);
  ~SiPixelClusterShapeCacheProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  using InputCollection = edmNew::DetSetVector<SiPixelCluster>;

  edm::EDGetTokenT<InputCollection> token_;
};

SiPixelClusterShapeCacheProducer::SiPixelClusterShapeCacheProducer(const edm::ParameterSet& iConfig):
  token_(consumes<InputCollection>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<SiPixelClusterShapeCache>();
}

SiPixelClusterShapeCacheProducer::~SiPixelClusterShapeCacheProducer() {}

void SiPixelClusterShapeCacheProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClusters"));
  descriptions.add("siPixelClusterShapeCache", desc);
}

void SiPixelClusterShapeCacheProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<InputCollection> input;
  iEvent.getByToken(token_, input);

  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get(geom);

  std::auto_ptr<SiPixelClusterShapeCache> output(new SiPixelClusterShapeCache(input));
  output->reserve(input->data().size());

  ClusterData data; // reused
  ClusterShape clusterShape;

  for(const auto& detSet: *input) {
    const GeomDetUnit *genericDet = geom->idToDetUnit(detSet.detId());
    const PixelGeomDetUnit *pixDet = dynamic_cast<const PixelGeomDetUnit *>(genericDet);
    assert(pixDet);

    edmNew::DetSet<SiPixelCluster>::const_iterator iCluster = detSet.begin(), endCluster = detSet.end();
    for(; iCluster != endCluster; ++iCluster) {
      clusterShape.determineShape(*pixDet, *iCluster, data);
      SiPixelClusterShapeCache::ClusterRef clusterRef = edmNew::makeRefTo(input, iCluster);
      output->push_back(clusterRef, data);
      data.size.clear();
    }
  }

  output->shrink_to_fit();
  iEvent.put(output);
}

DEFINE_FWK_MODULE(SiPixelClusterShapeCacheProducer);
