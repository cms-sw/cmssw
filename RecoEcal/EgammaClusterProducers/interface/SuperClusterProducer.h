#ifndef RecoEcal_EgammaClusterProducers_SuperClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_SuperClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "RecoEcal/EgammaClusterAlgos/interface/BremRecoveryClusterAlgo.h"

//


class SuperClusterProducer : public edm::EDProducer 
{
  
  public:

      SuperClusterProducer(const edm::ParameterSet& ps);

      ~SuperClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events
 
      BremRecoveryClusterAlgo::VerbosityLevel verbosity;

      std::string endcapClusterCollection_;
      std::string barrelClusterCollection_;

      std::string endcapClusterProducer_;
      std::string barrelClusterProducer_;

      std::string endcapSuperclusterCollection_;
      std::string barrelSuperclusterCollection_;

      float barrelEtaSearchRoad_;
      float barrelPhiSearchRoad_;
      float endcapEtaSearchRoad_; 
      float endcapPhiSearchRoad_;
      float seedTransverseEnergyThreshold_;

      bool doBarrel_;
      bool doEndcaps_;

      BremRecoveryClusterAlgo * bremAlgo_p;

      double totalE;
      int noSuperClusters;

      
      void getClusterPtrVector(edm::Event& evt, std::string clusterProducer_, std::string clusterCollection_, reco::CaloClusterPtrVector *);
  
      void produceSuperclustersForECALPart(edm::Event& evt, 
					   std::string clusterProducer, 
					   std::string clusterCollection,
					   std::string superclusterColection);

      void outputValidationInfo(reco::SuperClusterCollection &superclusterCollection);
    
      bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0));}
};


#endif
