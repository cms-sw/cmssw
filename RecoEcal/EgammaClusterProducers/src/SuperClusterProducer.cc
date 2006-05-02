// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"

// Reconstruction Classes
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "RecoEcal/EgammaClusterAlgos/interface/BremRecoveryClusterAlgo.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/SuperClusterProducer.h"


SuperClusterProducer::SuperClusterProducer(const edm::ParameterSet& ps)
{
  superclusterCollection_ = ps.getParameter<std::string>("superclusterCollection");
  produces< reco::SuperClusterCollection >(superclusterCollection_);
  nEvt_ = 0;
}


SuperClusterProducer::~SuperClusterProducer()
{
  std::cout << "Destructor of SuperClusterProducer called!" << std::endl;
}


void SuperClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // get the cluster collection out:
  edm::Handle<reco::BasicClusterCollection> bccHandle;
  evt.getByType(bccHandle);
  if (!(bccHandle.isValid())) 
    {
      std::cout << "could not get a handle on the BasicClusterCollection!" << std::endl;
      return;
    }
  reco::BasicClusterCollection clusterCollection = *bccHandle;
  std::cout << "Got the BasicClusterCollection" << std::endl;

  reco::BasicClusterRefVector clusterRefVector;

  for (unsigned int i = 0; i < clusterCollection.size(); i++)
    {
      clusterRefVector.push_back(reco::BasicClusterRef(bccHandle, i));
    }

  BremRecoveryClusterAlgo myBremRecovery;

  std::auto_ptr< reco::SuperClusterCollection > 
    superclusters_ap( new reco::SuperClusterCollection( myBremRecovery.makeSuperClusters(clusterRefVector) ) );

  evt.put(superclusters_ap, superclusterCollection_);

  std::cout << "SuperClusterCollection added to the Event! :-)" << std::endl;

  nEvt_++;
}

