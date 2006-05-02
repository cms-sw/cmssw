// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/IslandClusterProducer.h"


IslandClusterProducer::IslandClusterProducer(const edm::ParameterSet& ps)
{
  island_p = new IslandClusterAlgo(ps.getParameter<double>("IslandBarrelSeedThr"), 
				   ps.getParameter<double>("IslandEndcapSeedThr"));

  clusterCollection_ = ps.getParameter<std::string>("clusterCollection");
  produces< reco::BasicClusterCollection >(clusterCollection_);
  nEvt_ = 0;
}


IslandClusterProducer::~IslandClusterProducer()
{
}


void IslandClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByType(rhcHandle);
  if (!(rhcHandle.isValid())) 
    {
      std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
      return;
    }
  EcalRecHitCollection hit_collection = *rhcHandle;

  // get the barrel geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *geometry_p = (*geoHandle).getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  CaloSubdetectorGeometry geometry = *geometry_p;

  // make the clusters!
  reco::BasicClusterCollection clusters = island_p->makeClusters(hit_collection, geometry);
  std::cout << "Finished clustering - BasicClusterCollection returned to producer..." << std::endl;

  // create an auto_ptr to a BasicClusterCollection, copy the clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > clusters_p(new reco::BasicClusterCollection);
  clusters_p->assign(clusters.begin(), clusters.end());
  evt.put(clusters_p, clusterCollection_);

  std::cout << "BasicClusterCollection added to the Event! :-)" << std::endl;

  nEvt_++;
}

