// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

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
  hitProducer_       = ps.getParameter<std::string>("hitProducer");
  hitCollection_     = ps.getParameter<std::string>("hitCollection");
  clusterCollection_ = ps.getParameter<std::string>("clusterCollection");
  double barrelSeedThreshold = ps.getUntrackedParameter<double>("IslandBarrelSeedThr", 2);
  double endcapSeedThreshold = ps.getUntrackedParameter<double>("IslandEndcapSeedThr", 2);

  produces< reco::BasicClusterCollection >(clusterCollection_);

  island_p = new IslandClusterAlgo(barrelSeedThreshold, endcapSeedThreshold);

  nEvt_ = 0;
}


IslandClusterProducer::~IslandClusterProducer()
{
  delete island_p;
}


void IslandClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;
  try
    {
      evt.getByLabel(hitProducer_, hitCollection_, rhcHandle);
      if (!(rhcHandle.isValid())) 
	{
	  std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
	  return;
	}
    }
  catch ( cms::Exception& ex ) 
    {
      edm::LogError("IslandClusterProducerError") << "Error! can't get the product " << hitCollection_.c_str() ;
    }

  const EcalRecHitCollection *hitCollection_p = rhcHandle.product();

  // get the barrel geometry from the event setup:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<IdealGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);

  // Run the clusterization algorithm:
  reco::BasicClusterCollection clusters;
  clusters = island_p->makeClusters(*hitCollection_p, geometry_p);

  edm::LogInfo("IslandClusterProducerInfo") 
    << "Finished clustering - BasicClusterCollection returned to producer..." 
    << std::endl;

  // create an auto_ptr to a BasicClusterCollection, copy the clusters into it and put in the Event:
  std::auto_ptr< reco::BasicClusterCollection > clusters_p(new reco::BasicClusterCollection);
  clusters_p->assign(clusters.begin(), clusters.end());
  evt.put(clusters_p, clusterCollection_);

  edm::LogInfo("IslandClusterProducerInfo") 
    << "BasicClusterCollection added to the Event! :-)" 
    << std::endl;

  nEvt_++;
}

