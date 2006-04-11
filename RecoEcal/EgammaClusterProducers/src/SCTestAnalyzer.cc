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
#include "RecoEcal/EgammaClusterProducers/interface/SCTestAnalyzer.h"
#include "RecoEcal/EgammaClusterAlgos/interface/BremRecoveryClusterAlgo.h"
#include "RecoEcal/EgammaClusterAlgos/interface/SuperCluster.h"

SCTestAnalyzer::SCTestAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


SCTestAnalyzer::~SCTestAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


void SCTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::cout << "I am about to get the BasicClusterCollection from the event" << std::endl;

  // get the cluster collection out:
  edm::Handle<reco::BasicClusterCollection> bccHandle;
  iEvent.getByType(bccHandle);
  if (!(bccHandle.isValid())) 
    {
      std::cout << "could not get a handle on the BasicClusterCollection!" << std::endl;
      return;
    }
  reco::BasicClusterCollection cluster_collection = *bccHandle;

  std::cout << "Got the BasicClusterCollection" << std::endl;

  BremRecoveryClusterAlgo myBremRecovery;

  std::vector<SuperCluster> mySuperClusters = myBremRecovery.makeSuperClusters(cluster_collection);
}

