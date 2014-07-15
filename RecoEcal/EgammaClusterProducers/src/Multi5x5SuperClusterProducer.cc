// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// Reconstruction Classes
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/Multi5x5SuperClusterProducer.h"


Multi5x5SuperClusterProducer::Multi5x5SuperClusterProducer(const edm::ParameterSet& ps)
{

  eeClustersToken_ = 
	  consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("endcapClusterTag"));
  ebClustersToken_ = 
	  consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("barrelClusterTag"));


  endcapSuperclusterCollection_ = ps.getParameter<std::string>("endcapSuperclusterCollection");
  barrelSuperclusterCollection_ = ps.getParameter<std::string>("barrelSuperclusterCollection");

  doBarrel_ = ps.getParameter<bool>("doBarrel");
  doEndcaps_ = ps.getParameter<bool>("doEndcaps");

  barrelEtaSearchRoad_ = ps.getParameter<double>("barrelEtaSearchRoad");
  barrelPhiSearchRoad_ = ps.getParameter<double>("barrelPhiSearchRoad");
  endcapEtaSearchRoad_ = ps.getParameter<double>("endcapEtaSearchRoad");
  endcapPhiSearchRoad_ = ps.getParameter<double>("endcapPhiSearchRoad");
  seedTransverseEnergyThreshold_ = ps.getParameter<double>("seedTransverseEnergyThreshold");

  const edm::ParameterSet bremRecoveryPset = ps.getParameter<edm::ParameterSet>("bremRecoveryPset");
  bool dynamicPhiRoad = ps.getParameter<bool>("dynamicPhiRoad");

  bremAlgo_p = new Multi5x5BremRecoveryClusterAlgo(bremRecoveryPset, barrelEtaSearchRoad_, barrelPhiSearchRoad_, 
					 endcapEtaSearchRoad_, endcapPhiSearchRoad_, 
                                         dynamicPhiRoad, seedTransverseEnergyThreshold_);

  produces< reco::SuperClusterCollection >(endcapSuperclusterCollection_);
  produces< reco::SuperClusterCollection >(barrelSuperclusterCollection_);

  totalE = 0;
  noSuperClusters = 0;
  nEvt_ = 0;
}


Multi5x5SuperClusterProducer::~Multi5x5SuperClusterProducer()
{
  delete bremAlgo_p;
}

void
Multi5x5SuperClusterProducer::endStream() {
  double averEnergy = 0.;
  std::ostringstream str;
  str << "Multi5x5SuperClusterProducer::endJob()\n"
      << "  total # reconstructed super clusters: " << noSuperClusters << "\n"
      << "  total energy of all clusters: " << totalE << "\n";
  if(noSuperClusters>0) { 
    averEnergy = totalE / noSuperClusters;
    str << "  average SuperCluster energy = " << averEnergy << "\n";
  }
  edm::LogInfo("Multi5x5SuperClusterProducerInfo") << str.str() << "\n";

}


void Multi5x5SuperClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if(doEndcaps_)
    produceSuperclustersForECALPart(evt, eeClustersToken_, endcapSuperclusterCollection_);

  if(doBarrel_)
    produceSuperclustersForECALPart(evt, ebClustersToken_, barrelSuperclusterCollection_);

  nEvt_++;
}


void Multi5x5SuperClusterProducer::
produceSuperclustersForECALPart(edm::Event& evt, 
								const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken,
								std::string superclusterCollection)
{
  // get the cluster collection out and turn it to a BasicClusterRefVector:
  reco::CaloClusterPtrVector *clusterPtrVector_p = new reco::CaloClusterPtrVector;
  getClusterPtrVector(evt, clustersToken, clusterPtrVector_p);

  // run the brem recovery and get the SC collection
  std::auto_ptr<reco::SuperClusterCollection> 
    superclusters_ap(new reco::SuperClusterCollection(bremAlgo_p->makeSuperClusters(*clusterPtrVector_p)));

  // count the total energy and the number of superclusters
  reco::SuperClusterCollection::iterator it;
  for (it = superclusters_ap->begin(); it != superclusters_ap->end(); it++)
    {
      totalE += it->energy();
      noSuperClusters++;
    }

  // put the SC collection in the event
  evt.put(superclusters_ap, superclusterCollection);

  delete clusterPtrVector_p;
}


void Multi5x5SuperClusterProducer::getClusterPtrVector(edm::Event& evt, 
													   const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken, 
													   reco::CaloClusterPtrVector *clusterPtrVector_p)
{  
  edm::Handle<reco::BasicClusterCollection> bccHandle;
  evt.getByToken(clustersToken, bccHandle);
  
  const reco::BasicClusterCollection *clusterCollection_p = bccHandle.product();
  for (unsigned int i = 0; i < clusterCollection_p->size(); i++)
    {
      clusterPtrVector_p->push_back(reco::CaloClusterPtr(bccHandle, i));
    }
}                               
