// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// Reconstruction Classes
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/SuperClusterProducer.h"


SuperClusterProducer::SuperClusterProducer(const edm::ParameterSet& ps)
{
  endcapClusterProducer_ = ps.getParameter<std::string>("endcapClusterProducer");
  barrelClusterProducer_ = ps.getParameter<std::string>("barrelClusterProducer");

  endcapClusterCollection_ = ps.getParameter<std::string>("endcapClusterCollection");
  barrelClusterCollection_ = ps.getParameter<std::string>("barrelClusterCollection");

  endcapSuperclusterCollection_ = ps.getParameter<std::string>("endcapSuperclusterCollection");
  barrelSuperclusterCollection_ = ps.getParameter<std::string>("barrelSuperclusterCollection");

  barrelEtaSearchRoad_ = ps.getParameter<double>("barrelEtaSearchRoad");
  barrelPhiSearchRoad_ = ps.getParameter<double>("barrelPhiSearchRoad");
  endcapEtaSearchRoad_ = ps.getParameter<double>("endcapEtaSearchRoad");
  endcapPhiSearchRoad_ = ps.getParameter<double>("endcapPhiSearchRoad");
  seedEnergyThreshold_ = ps.getParameter<double>("barrelEtaSearchRoad");

  bremAlgo_p = new BremRecoveryClusterAlgo(barrelEtaSearchRoad_, barrelPhiSearchRoad_, 
					 endcapEtaSearchRoad_, endcapPhiSearchRoad_, 
					 seedEnergyThreshold_);

  produces< reco::SuperClusterCollection >(endcapSuperclusterCollection_);
  produces< reco::SuperClusterCollection >(barrelSuperclusterCollection_);

  totalE = 0;
  noSuperClusters = 0;
  nEvt_ = 0;
}


SuperClusterProducer::~SuperClusterProducer()
{
  double averEnergy = totalE / noSuperClusters;
  std::cout << "----------------------------------------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------------------------------------" << std::endl;
  std::cout << "SuperClusterProducerInfo: " << "average SuperCluster energy = " << averEnergy << std::endl;
  std::cout << "----------------------------------------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------------------------------------" << std::endl;
  delete bremAlgo_p;
}


void SuperClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  produceSuperclustersForECALPart(evt, endcapClusterProducer_, endcapClusterCollection_, endcapSuperclusterCollection_);
  produceSuperclustersForECALPart(evt, barrelClusterProducer_, barrelClusterCollection_, barrelSuperclusterCollection_);

  /*
  // get the cluster collections out and turn them to BasicClusterRefVectors:
  reco::BasicClusterRefVector *endcapClusterRefVector = getClusterRefVector(evt, endcapClusterProducer_, endcapClusterCollection_);
  reco::BasicClusterRefVector *barrelClusterRefVector = getClusterRefVector(evt, barrelClusterProducer_, barrelClusterCollection_);

  // run the brem recovery and get the SC collections
  reco::SuperClusterCollection *
    endcapSCCollection_p = new reco::SuperClusterCollection(bremAlgo_p->makeSuperClusters(*endcapClusterRefVector));
  reco::SuperClusterCollection *
    barrelSCCollection_p = new reco::SuperClusterCollection(bremAlgo_p->makeSuperClusters(*barrelClusterRefVector));

  std::auto_ptr<reco::SuperClusterCollection> endcapSuperclusters_ap(new reco::SuperClusterCollection);
  endcapSuperclusters_ap->assign(endcapSCCollection_p->begin(), endcapSCCollection_p->end());
  evt.put(endcapSuperclusters_ap, endcapSuperclusterCollection_);

  std::auto_ptr<reco::SuperClusterCollection> barrelSuperclusters_ap(new reco::SuperClusterCollection);
  barrelSuperclusters_ap->assign(barrelSCCollection_p->begin(), barrelSCCollection_p->end());
  evt.put(barrelSuperclusters_ap, barrelSuperclusterCollection_);
  */
  nEvt_++;
}

void SuperClusterProducer::produceSuperclustersForECALPart(edm::Event& evt, 
							   std::string clusterProducer, 
							   std::string clusterCollection,
							   std::string superclusterCollection)
{
  // get the cluster collection out and turn it to a BasicClusterRefVector:
  reco::BasicClusterRefVector *clusterRefVector = getClusterRefVector(evt, clusterProducer, clusterCollection);

  // run the brem recovery and get the SC collections
  reco::SuperClusterCollection *
    superclusterCollection_p = new reco::SuperClusterCollection(bremAlgo_p->makeSuperClusters(*clusterRefVector));

  std::auto_ptr<reco::SuperClusterCollection> superclusters_ap(new reco::SuperClusterCollection);
  superclusters_ap->assign(superclusterCollection_p->begin(), superclusterCollection_p->end());
  evt.put(superclusters_ap, superclusterCollection);

  reco::SuperClusterCollection::iterator it;
  for (it = superclusterCollection_p->begin(); it != superclusterCollection_p->end(); it++)
    {
      totalE += it->energy();
      noSuperClusters++;
    }
}


reco::BasicClusterRefVector *
SuperClusterProducer::getClusterRefVector(edm::Event& evt, std::string clusterProducer_, std::string clusterCollection_)
{  
  edm::Handle<reco::BasicClusterCollection> bccHandle;
  try
    {
      evt.getByLabel(clusterProducer_, clusterCollection_, bccHandle);
      if (!(bccHandle.isValid()))
	{
	  edm::LogError("SuperClusterProducerError") << "could not get a handle on the BasicCluster Collection!" << std::endl;
	  return 0;
	}
    } 
  catch ( cms::Exception& ex )
    {
      edm::LogError("SuperClusterProducerError") << "Error! can't get the product " << clusterCollection_.c_str() ; 
      return 0;
    }

  const reco::BasicClusterCollection *clusterCollection_p = bccHandle.product();
  reco::BasicClusterRefVector *clusterRefVector_p = new reco::BasicClusterRefVector;
  for (unsigned int i = 0; i < clusterCollection_p->size(); i++)
    {
      clusterRefVector_p->push_back(reco::BasicClusterRef(bccHandle, i));
    }

  return clusterRefVector_p;
}                               




