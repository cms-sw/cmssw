#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"

#include <vector>
#include <string>

class Phase2TrackerRecHits : public edm::global::EDProducer<> {
public:
  explicit Phase2TrackerRecHits(const edm::ParameterSet& conf);
  ~Phase2TrackerRecHits() override{};
  void produce(edm::StreamID sid, edm::Event& event, const edm::EventSetup& eventSetup) const final;

private:
  edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> token_;
  edm::ESInputTag cpeTag_;
};

Phase2TrackerRecHits::Phase2TrackerRecHits(edm::ParameterSet const& conf)
    : token_(consumes<Phase2TrackerCluster1DCollectionNew>(conf.getParameter<edm::InputTag>("src"))),
      cpeTag_(conf.getParameter<edm::ESInputTag>("Phase2StripCPE")) {
  produces<Phase2TrackerRecHit1DCollectionNew>();
}

void Phase2TrackerRecHits::produce(edm::StreamID sid, edm::Event& event, const edm::EventSetup& eventSetup) const {
  // Get the Clusters
  edm::Handle<Phase2TrackerCluster1DCollectionNew> clusters;
  event.getByToken(token_, clusters);

  // load the cpe via the eventsetup
  edm::ESHandle<ClusterParameterEstimator<Phase2TrackerCluster1D> > cpe;
  eventSetup.get<TkStripCPERecord>().get(cpeTag_, cpe);

  // Get the geometry
  edm::ESHandle<TrackerGeometry> geomHandle;
  eventSetup.get<TrackerDigiGeometryRecord>().get(geomHandle);
  const TrackerGeometry* tkGeom(&(*geomHandle));

  // Global container for the RecHits of each module
  auto outputRecHits = std::make_unique<Phase2TrackerRecHit1DCollectionNew>();

  // Loop over clusters
  for (const auto& clusterDetSet : *clusters) {
    DetId detId(clusterDetSet.detId());

    // Geometry
    const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));

    // Container for the clusters that will be produced for this modules
    Phase2TrackerRecHit1DCollectionNew::FastFiller rechits(*outputRecHits, clusterDetSet.detId());

    for (const auto& clusterRef : clusterDetSet) {
      ClusterParameterEstimator<Phase2TrackerCluster1D>::LocalValues lv =
          cpe->localParameters(clusterRef, *geomDetUnit);

      // Create a persistent edm::Ref to the cluster
      edm::Ref<Phase2TrackerCluster1DCollectionNew, Phase2TrackerCluster1D> cluster =
          edmNew::makeRefTo(clusters, &clusterRef);

      // Make a RecHit and add it to the DetSet
      Phase2TrackerRecHit1D hit(lv.first, lv.second, *geomDetUnit, cluster);

      rechits.push_back(hit);
    }
  }

  outputRecHits->shrink_to_fit();
  event.put(std::move(outputRecHits));
}

DEFINE_FWK_MODULE(Phase2TrackerRecHits);
