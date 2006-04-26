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
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/HybridClusterProducer.h"


HybridClusterProducer::HybridClusterProducer(const edm::ParameterSet& ps)
{
  hybrid_p = new HybridClusterAlgo(ps.getParameter<double>("HybridBarrelSeedThr"), 
				   ps.getParameter<double>("HybridEndcapSeedThr"),
				   ps.getParameter<int>("step"),
				   ps.getParameter<double>("ethresh"),
				   ps.getParameter<double>("ewing"),
				   ps.getParameter<double>("eseed"));

  basicclusterCollection_ = ps.getParameter<std::string>("basicclusterCollection");
  superclusterCollection_ = ps.getParameter<std::string>("superclusterCollection");
  produces< reco::BasicClusterCollection >(basicclusterCollection_);
  produces< reco::SuperClusterCollection >(superclusterCollection_);
  nEvt_ = 0;
}


HybridClusterProducer::~HybridClusterProducer()
{
}


void HybridClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
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

  // make the Basic clusters!
  reco::BasicClusterCollection basicClusters;
  hybrid_p->makeClusters(hit_collection, geoHandle, basicClusters);
  std::cout << "Finished clustering - BasicClusterCollection returned to producer..." << std::endl;

  // create an auto_ptr to a BasicClusterCollection, copy the clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > basicclusters_p(new reco::BasicClusterCollection);
  basicclusters_p->assign(basicClusters.begin(), basicClusters.end());
  evt.put(basicclusters_p, basicclusterCollection_);
  //Basic clusters now in the event.
  std::cout << "Basic Clusters now put into event." << std::endl;
  
  //Weird though it is, get the BasicClusters back out of the event.  We need the
  //edm::Ref to these guys to make our superclusters for Hybrid.
  edm::Handle<reco::BasicClusterCollection> bccHandle;
  evt.getByLabel(basicclusterCollection_, bccHandle);
  if (!(bccHandle.isValid())) {
    std::cout << "could not get a handle on the BasicClusterCollection!" << std::endl;
    return;
  }
  reco::BasicClusterCollection clusterCollection = *bccHandle;
  std::cout << "Got the BasicClusterCollection" << std::endl;

  reco::BasicClusterRefVector clusterRefVector;
  for (unsigned int i = 0; i < clusterCollection.size(); i++){
    clusterRefVector.push_back(reco::BasicClusterRef(bccHandle, i));
  }

  reco::SuperClusterCollection superClusters = hybrid_p->makeSuperClusters(clusterRefVector);
  
  std::auto_ptr< reco::SuperClusterCollection > superclusters_p(new reco::SuperClusterCollection);
  superclusters_p->assign(superClusters.begin(), superClusters.end());
  evt.put(superclusters_p, superclusterCollection_);

  std::cout << "Hybrid Clusters (Basic/Super) added to the Event! :-)" << std::endl;

  nEvt_++;
}

