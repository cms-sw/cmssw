// C/C++ headers
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>

// Framework
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
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
#include "RecoHI/HiEgammaAlgos/plugins/HiSuperClusterProducer.h"


HiSuperClusterProducer::HiSuperClusterProducer(const edm::ParameterSet& ps)
{
   // The verbosity level
   std::string verbosityString = ps.getParameter<std::string>("VerbosityLevel");
   if      (verbosityString == "DEBUG")   verbosity = HiBremRecoveryClusterAlgo::pDEBUG;
   else if (verbosityString == "WARNING") verbosity = HiBremRecoveryClusterAlgo::pWARNING;
   else if (verbosityString == "INFO")    verbosity = HiBremRecoveryClusterAlgo::pINFO;
   else                                   verbosity = HiBremRecoveryClusterAlgo::pERROR;


   endcapSuperclusterCollection_ = ps.getParameter<std::string>("endcapSuperclusterCollection");
   barrelSuperclusterCollection_ = ps.getParameter<std::string>("barrelSuperclusterCollection");

   doBarrel_ = ps.getParameter<bool>("doBarrel");
   doEndcaps_ = ps.getParameter<bool>("doEndcaps");


   barrelEtaSearchRoad_ = ps.getParameter<double>("barrelEtaSearchRoad");
   barrelPhiSearchRoad_ = ps.getParameter<double>("barrelPhiSearchRoad");
   endcapEtaSearchRoad_ = ps.getParameter<double>("endcapEtaSearchRoad");
   endcapPhiSearchRoad_ = ps.getParameter<double>("endcapPhiSearchRoad");
   seedTransverseEnergyThreshold_ = ps.getParameter<double>("seedTransverseEnergyThreshold");
   barrelBCEnergyThreshold_ = ps.getParameter<double>("barrelBCEnergyThreshold");
   endcapBCEnergyThreshold_ = ps.getParameter<double>("endcapBCEnergyThreshold");

   if (verbosityString == "INFO") {
      std::cout <<"Barrel BC Energy threshold = "<<barrelBCEnergyThreshold_<<std::endl;
      std::cout <<"Endcap BC Energy threshold = "<<endcapBCEnergyThreshold_<<std::endl;
   }

   bremAlgo_p = new HiBremRecoveryClusterAlgo(barrelEtaSearchRoad_, barrelPhiSearchRoad_, 
                                              endcapEtaSearchRoad_, endcapPhiSearchRoad_, 
                                              seedTransverseEnergyThreshold_,
                                              barrelBCEnergyThreshold_,
                                              endcapBCEnergyThreshold_,
                                              verbosity);

   produces< reco::SuperClusterCollection >(endcapSuperclusterCollection_);
   produces< reco::SuperClusterCollection >(barrelSuperclusterCollection_);

   eeClustersToken_ =  consumes<reco::BasicClusterCollection>(edm::InputTag(ps.getParameter<std::string>("endcapClusterProducer"), 
									    ps.getParameter<std::string>("endcapClusterCollection")));
   ebClustersToken_ =  consumes<reco::BasicClusterCollection>(edm::InputTag(ps.getParameter<std::string>("barrelClusterProducer"), 
									    ps.getParameter<std::string>("barrelClusterCollection")));

   totalE = 0;
   noSuperClusters = 0;
   nEvt_ = 0;
}


HiSuperClusterProducer::~HiSuperClusterProducer()
{
   delete bremAlgo_p;
}

void HiSuperClusterProducer::endJob() {
   double averEnergy = 0.;
   std::ostringstream str;
   str << "HiSuperClusterProducer::endJob()\n"
       << "  total # reconstructed super clusters: " << noSuperClusters << "\n"
       << "  total energy of all clusters: " << totalE << "\n";
   if(noSuperClusters>0) { 
     averEnergy = totalE / noSuperClusters;
     str << "  average SuperCluster energy = " << averEnergy << "\n";
   }
   edm::LogInfo("HiSuperClusterProducerInfo") << str.str() << "\n";
 
}


void HiSuperClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if(doEndcaps_)
    produceSuperclustersForECALPart(evt, eeClustersToken_, endcapSuperclusterCollection_);

  if(doBarrel_)
    produceSuperclustersForECALPart(evt, ebClustersToken_, barrelSuperclusterCollection_);

  nEvt_++;
}


void HiSuperClusterProducer::produceSuperclustersForECALPart(edm::Event& evt, 
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


void HiSuperClusterProducer::getClusterPtrVector(edm::Event& evt, const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken, reco::CaloClusterPtrVector *clusterPtrVector_p)
{  
  edm::Handle<reco::BasicClusterCollection> bccHandle;

  evt.getByToken(clustersToken, bccHandle);

  if (!(bccHandle.isValid()))
    {
      edm::LogError("HiSuperClusterProducerError") << "could not get a handle on the BasicCluster Collection!";
      clusterPtrVector_p = 0;
    }

  const reco::BasicClusterCollection *clusterCollection_p = bccHandle.product();
  for (unsigned int i = 0; i < clusterCollection_p->size(); i++)
    {
      clusterPtrVector_p->push_back(reco::CaloClusterPtr(bccHandle, i));
    }
}                               



//define this as a plug-in                                                                                                                                                   
DEFINE_FWK_MODULE(HiSuperClusterProducer);
