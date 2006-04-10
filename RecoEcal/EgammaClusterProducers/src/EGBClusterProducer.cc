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
#include "RecoEcal/EgammaClusterProducers/interface/EGBClusterProducer.h"

EGBClusterProducer::EGBClusterProducer(const edm::ParameterSet& conf)
{
  island_p = new IslandClusterAlgo(conf.getParameter<double>("IslandBarrelSeedThr"), 
			conf.getParameter<double>("IslandEndcapSeedThr"));

  hybrid_p = new HybridClusterAlgo(conf.getParameter<double>("EGBasicHybridSeedThr"),
			conf.getParameter<double>("EGBasicHybridSeedThrEndcap"), // correct in the algorithm first, then here
			conf.getParameter<int>("EGBasicHybridNstep"),
			conf.getParameter<double>("EGBasicHybridEGBasicHybridEdomino"),
			conf.getParameter<double>("EGBasicHybridEsubCluster"),
			conf.getParameter<double>("EGBasicHybridEwing"));
  
  //  logWeightedPosition_(conf.getParameter<double>("logWeightedPosition"))
  
  produces<reco::BasicClusterCollection>();
  //positionCalculator_ = new EGBClusPosCalculator();
}

EGBClusterProducer::~EGBClusterProducer()
{
  //delete positionCalculator_;
}

// ------------ method called to produce the data  ------------
void
EGBClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByType(rhcHandle);
  if (!(rhcHandle.isValid())) 
    {
      std::cout << "could not get handle on the event!" << std::endl;
      return;
    }
  EcalRecHitCollection hit_collection = *rhcHandle;
  
  // get the barrel geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *geometry_p = (*geoHandle).getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  CaloSubdetectorGeometry const geometry = *geometry_p;
  
  //Use the EcalRecHitCollection to create an EGBClusterCollection which 
  //is put into the Event
  // make the clusters
  std::vector<reco::BasicCluster> islands = island_p->makeClusters(hit_collection, geometry);
  std::vector<reco::BasicCluster> hybrids = hybrid_p->makeClusters(hit_collection, geometry);

  std::auto_ptr<reco::BasicClusterCollection> clusters_p(new reco::BasicClusterCollection);

  clusters_p->insert(clusters_p->end(),islands.begin(),islands.end());
  clusters_p->insert(clusters_p->end(),hybrids.begin(),hybrids.end());

   /*
   std::auto_ptr<EGBClusterCollection>::iterator it = result->begin();;
   if (logWeightedPosition_) {
     for(; it != result->end(); it++)
       positionCalculator_->correct(**it);
   }
   */

  evt.put(clusters_p);
  std::cout << "BasicClusterCollection added to the Event! :-)" << std::endl;

}

