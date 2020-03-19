#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#ifdef VERIFY_PH2_TK_CLUS
#include "Phase2TrackerClusterizerAlgorithm.h"
#endif
#include "Phase2TrackerClusterizerSequentialAlgorithm.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <memory>

class Phase2TrackerClusterizer : public edm::stream::EDProducer<> {
public:
  explicit Phase2TrackerClusterizer(const edm::ParameterSet& conf);
  ~Phase2TrackerClusterizer() override;
  void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

private:
#ifdef VERIFY_PH2_TK_CLUS
  std::unique_ptr<Phase2TrackerClusterizerAlgorithm> clusterizer_;
#endif
  edm::EDGetTokenT<edm::DetSetVector<Phase2TrackerDigi> > token_;
};

/*
     * Initialise the producer
     */

Phase2TrackerClusterizer::Phase2TrackerClusterizer(edm::ParameterSet const& conf)
    :
#ifdef VERIFY_PH2_TK_CLUS
      clusterizer_(new Phase2TrackerClusterizerAlgorithm(conf.getParameter<unsigned int>("maxClusterSize"),
                                                         conf.getParameter<unsigned int>("maxNumberClusters"))),
#endif
      token_(consumes<edm::DetSetVector<Phase2TrackerDigi> >(conf.getParameter<edm::InputTag>("src"))) {
  produces<Phase2TrackerCluster1DCollectionNew>();
}

Phase2TrackerClusterizer::~Phase2TrackerClusterizer() {}

/*
     * Clusterize the events
     */

void Phase2TrackerClusterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  // Get the Digis
  edm::Handle<edm::DetSetVector<Phase2TrackerDigi> > digis;
  event.getByToken(token_, digis);

#ifdef VERIFY_PH2_TK_CLUS
  // Get the geometry
  edm::ESHandle<TrackerGeometry> geomHandle;
  eventSetup.get<TrackerDigiGeometryRecord>().get(geomHandle);
  const TrackerGeometry* tkGeom(&(*geomHandle));
  // Global container for the clusters of each modules
  auto outputClustersOld = std::make_unique<Phase2TrackerCluster1DCollectionNew>();
#endif
  auto outputClusters = std::make_unique<Phase2TrackerCluster1DCollectionNew>();

  // Go over all the modules
  for (auto DSViter : *digis) {
    DetId detId(DSViter.detId());

    Phase2TrackerCluster1DCollectionNew::FastFiller clusters(*outputClusters, DSViter.detId());
    Phase2TrackerClusterizerSequentialAlgorithm algo;
    algo.clusterizeDetUnit(DSViter, clusters);
    if (clusters.empty())
      clusters.abort();

#ifdef VERIFY_PH2_TK_CLUS
    if (!clusters.empty()) {
      auto cp = clusters[0].column();
      auto sp = clusters[0].firstStrip();
      for (auto const& cl : clusters) {
        if (cl.column() < cp)
          std::cout << "column not in order! " << std::endl;
        if (cl.column() == cp && cl.firstStrip() < sp)
          std::cout << "strip not in order! " << std::endl;
        cp = cl.column();
        sp = cl.firstStrip();
      }
    }
#endif

#ifdef VERIFY_PH2_TK_CLUS
    // Geometry
    const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(geomDetUnit);
    if (!pixDet)
      assert(0);

    // Container for the clusters that will be produced for this modules
    Phase2TrackerCluster1DCollectionNew::FastFiller clustersOld(*outputClustersOld, DSViter.detId());

    // Setup the clusterizer algorithm for this detector (see ClusterizerAlgorithm for more details)
    clusterizer_->setup(pixDet);

    // Pass the list of Digis to the main algorithm
    // This function will store the clusters in the previously created container
    clusterizer_->clusterizeDetUnit(DSViter, clustersOld);
    if (clustersOld.empty())
      clustersOld.abort();

    if (clusters.size() != clustersOld.size()) {
      std::cout << "SIZEs " << int(detId) << ' ' << clusters.size() << ' ' << clustersOld.size() << std::endl;
      for (auto const& cl : clusters)
        std::cout << cl.size() << ' ' << cl.threshold() << ' ' << cl.firstRow() << ' ' << cl.column() << std::endl;
      std::cout << "Old " << std::endl;
      for (auto const& cl : clustersOld)
        std::cout << cl.size() << ' ' << cl.threshold() << ' ' << cl.firstRow() << ' ' << cl.column() << std::endl;
    }
#endif
  }

#ifdef VERIFY_PH2_TK_CLUS
  // std::cout << "SIZEs " << outputClusters->dataSize() << ' ' << outputClustersOld->dataSize() << std::endl;
  assert(outputClusters->dataSize() == outputClustersOld->dataSize());
  for (auto i = 0U; i < outputClusters->dataSize(); ++i) {
    assert(outputClusters->data()[i].size() == outputClustersOld->data()[i].size());
    assert(outputClusters->data()[i].threshold() == outputClustersOld->data()[i].threshold());
    assert(outputClusters->data()[i].firstRow() == outputClustersOld->data()[i].firstRow());
    assert(outputClusters->data()[i].column() == outputClustersOld->data()[i].column());
  }
#endif

  // Add the data to the output
  outputClusters->shrink_to_fit();
  event.put(std::move(outputClusters));
}

DEFINE_FWK_MODULE(Phase2TrackerClusterizer);
