// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>
#include <sstring>

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
  // The verbosity level
  std::string verbosityString = ps.getParameter<std::string>("VerbosityLevel");
  if      (verbosityString == "DEBUG")   verbosity = BremRecoveryClusterAlgo::pDEBUG;
  else if (verbosityString == "WARNING") verbosity = BremRecoveryClusterAlgo::pWARNING;
  else if (verbosityString == "INFO")    verbosity = BremRecoveryClusterAlgo::pINFO;
  else                                   verbosity = BremRecoveryClusterAlgo::pERROR;

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
  seedEnergyThreshold_ = ps.getParameter<double>("seedEnergyThreshold");

  bremAlgo_p = new BremRecoveryClusterAlgo(barrelEtaSearchRoad_, barrelPhiSearchRoad_, 
					 endcapEtaSearchRoad_, endcapPhiSearchRoad_, 
					 seedEnergyThreshold_, verbosity);

  produces< reco::SuperClusterCollection >(endcapSuperclusterCollection_);
  produces< reco::SuperClusterCollection >(barrelSuperclusterCollection_);

  totalE = 0;
  noSuperClusters = 0;
  nEvt_ = 0;
}


SuperClusterProducer::~SuperClusterProducer()
{
  delete bremAlgo_p;
}

void
SuperClusterProducer::endJob() {
  double averEnergy = totalE / noSuperClusters;
  std::ostringstream str;
  str << "-------------------------------------------------------\n";
  str << "-------------------------------------------------------\n";
  str << "average SuperCluster energy = " << averEnergy << "\n";
  str << "-------------------------------------------------------\n";
  str << "-------------------------------------------------------\n";

  edm::LogInfo("SuperClusterProducerInfo") << str.str() << "\n";

}


void SuperClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  produceSuperclustersForECALPart(evt, endcapClusterProducer_, endcapClusterCollection_, endcapSuperclusterCollection_);
  produceSuperclustersForECALPart(evt, barrelClusterProducer_, barrelClusterCollection_, barrelSuperclusterCollection_);

  nEvt_++;
}


void SuperClusterProducer::produceSuperclustersForECALPart(edm::Event& evt, 
							   std::string clusterProducer, 
							   std::string clusterCollection,
							   std::string superclusterCollection)
{
  // get the cluster collection out and turn it to a BasicClusterRefVector:
  reco::BasicClusterRefVector *clusterRefVector = getClusterRefVector(evt, clusterProducer, clusterCollection);

  // run the brem recovery and get the SC collection
  std::auto_ptr<reco::SuperClusterCollection> 
    superclusters_ap(new reco::SuperClusterCollection(bremAlgo_p->makeSuperClusters(*clusterRefVector)));

  // count the total energy and the number of superclusters
  reco::SuperClusterCollection::iterator it;
  for (it = superclusters_ap->begin(); it != superclusters_ap->end(); it++)
    {
      totalE += it->energy();
      noSuperClusters++;
    }

  // put the SC collection in the event
  evt.put(superclusters_ap, superclusterCollection);
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
	  edm::LogError("SuperClusterProducerError") << "could not get a handle on the BasicCluster Collection!";
	  return 0;
	}
    } 
  catch ( cms::Exception& ex )
    {
      edm::LogError("SuperClusterProducerError") << "Error! can't get the product " << clusterCollection_.c_str(); 
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




