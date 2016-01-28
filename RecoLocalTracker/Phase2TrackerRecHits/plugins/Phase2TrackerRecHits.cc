#include "RecoLocalTracker/Phase2TrackerRecHits/plugins/Phase2TrackerRecHits.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Phase2TrackerRecHit/interface/Phase2TrackerRecHit1D.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <vector>
#include <string>


Phase2TrackerRecHits::Phase2TrackerRecHits(edm::ParameterSet const& conf) : 
  token_(consumes< Phase2TrackerCluster1DCollectionNew >(conf.getParameter<edm::InputTag>("src"))),
  cpeTag_(conf.getParameter<edm::ESInputTag>("Phase2StripCPE")) {
    produces<Phase2TrackerRecHit1DCollectionNew>();
}


void Phase2TrackerRecHits::produce(edm::StreamID sid, edm::Event& event, const edm::EventSetup& eventSetup) const {

  // Get the Clusters
  edm::Handle< Phase2TrackerCluster1DCollectionNew > clusters;
  event.getByToken(token_, clusters);

  // load the cpe via the eventsetup
  edm::ESHandle<ClusterParameterEstimator<Phase2TrackerCluster1D> > cpe;
  eventSetup.get<TkStripCPERecord>().get(cpeTag_, cpe);

  // Get the geometry
  edm::ESHandle< TrackerGeometry > geomHandle;
  eventSetup.get< TrackerDigiGeometryRecord >().get(geomHandle);
  const TrackerGeometry* tkGeom(&(*geomHandle));

  edm::ESHandle< TrackerTopology > tTopoHandle;
  eventSetup.get< IdealGeometryRecord >().get(tTopoHandle);
  //const TrackerTopology* tTopo(tTopoHandle.product());

  // Global container for the RecHits of each module
  std::auto_ptr< Phase2TrackerRecHit1DCollectionNew > outputRecHits(new Phase2TrackerRecHit1DCollectionNew());

  // Loop over clusters
  for (edmNew::DetSetVector< Phase2TrackerCluster1D >::const_iterator DSViter = clusters->begin(); DSViter != clusters->end(); ++DSViter) {

    DetId detId(DSViter->detId());

    // Geometry
    const GeomDetUnit * geomDetUnit(tkGeom->idToDetUnit(detId));

    // Container for the clusters that will be produced for this modules
    Phase2TrackerRecHit1DCollectionNew::FastFiller rechits(*outputRecHits, DSViter->detId());

    for (edmNew::DetSet< Phase2TrackerCluster1D >::const_iterator clustIt = DSViter->begin(); clustIt != DSViter->end(); ++clustIt) {
      ClusterParameterEstimator<Phase2TrackerCluster1D>::LocalValues lv = cpe->localParameters(*clustIt, *geomDetUnit);

      // Create a persistent edm::Ref to the cluster
      edm::Ref< edmNew::DetSetVector< Phase2TrackerCluster1D >, Phase2TrackerCluster1D > cluster = edmNew::makeRefTo(clusters, clustIt);

      // Make a RecHit and add it to the DetSet
      Phase2TrackerRecHit1D hit(lv.first, lv.second, cluster);

      rechits.push_back(hit);
    }
  }

  outputRecHits->shrink_to_fit();
  event.put(outputRecHits);

}

DEFINE_FWK_MODULE(Phase2TrackerRecHits);
