/*! \class TTClusterBuilderNew
 *  \brief Plugin to create Tracker clusters for the Track-Trigger (TTClusters)
 *         from digis.
 *         This replaces the old TTClusterBuilder class. The algo it uses
 *         corresponds to that in the FE chips.
 *         (It is based on the code of the offline cluster producer
 *          Phase2TrackerCluserizer).
 *
 *  \author Ian Tomalin
 *  \date Aug. 2026
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/InputTag.h"

//#include "L1Trigger/TrackTrigger/interface/Phase2TrackerClusterizerSequentialAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithmNew_official.h"

#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"

#include <vector>
#include <memory>

class TTClusterBuilderNew : public edm::stream::EDProducer<> {
public:
  explicit TTClusterBuilderNew(const edm::ParameterSet& conf);
  ~TTClusterBuilderNew() override = default;
  void produce(edm::Event& event, const edm::EventSetup& iSetup) override;

private:
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken;  
  edm::EDGetTokenT<edm::DetSetVector<Phase2TrackerDigi> > token_;

  unsigned int maxClusterWidth_;
  bool enableClusterVetoes_;
};

/*
     * Initialise the producer
     */

TTClusterBuilderNew::TTClusterBuilderNew(edm::ParameterSet const& conf) {
  tTopoToken = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  tGeomToken = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  token_ = consumes<edm::DetSetVector<Phase2TrackerDigi> >(conf.getParameter<edm::InputTag>("src"));
  // IRT CHECK: TrackerDTC uses a Token here. Why doesn't this?
  produces<TTClusterDetSetVec>("ClusterInclusive");
  maxClusterWidth_ = conf.getParameter<unsigned int>("maxClusterWidth");
  enableClusterVetoes_ = conf.getParameter<bool>("enableClusterVetoes");
}

/*
     * Clusterize the events
     */

void TTClusterBuilderNew::produce(edm::Event& event, const edm::EventSetup& iSetup) {

  // Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &iSetup.getData(tTopoToken);
  const TrackerGeometry* const tGeom = &iSetup.getData(tGeomToken);

  // Get the Digis
  edm::Handle<edm::DetSetVector<Phase2TrackerDigi> > digis;
  event.getByToken(token_, digis);

  auto outputClusters = std::make_unique<TTClusterDetSetVec>();
  
  // Loop over the tracker modules
  for (const auto& DSViter : *digis) {
    const DetId detId(DSViter.detId());
    const bool upperSensor = not tTopo->isLower(detId);
    const bool isPSp = (tGeom->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP);    

    // Define utility for adding clusters to output collection.
    TTClusterDetSetVec::FastFiller clusters(*outputClusters, DSViter.detId());
    
    // Create & run clustering algorithm
    //const Phase2TrackerClusterizerSequentialAlgorithm algo(maxClusterWidth_, detId, upperSensor, isPSp);
    TTClusterAlgorithmNew_official algo(maxClusterWidth_, enableClusterVetoes_, detId, upperSensor, isPSp);    
    
    algo.clusterizeDetUnit(DSViter, clusters);
    if (clusters.empty())
      clusters.abort();
  }

  // Add the data to the output
  outputClusters->shrink_to_fit();
  event.put(std::move(outputClusters), "ClusterInclusive");
}

DEFINE_FWK_MODULE(TTClusterBuilderNew);
