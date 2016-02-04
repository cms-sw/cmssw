#ifndef RecoEcal_EgammaClusterProducers_Multi5x5SuperClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_Multi5x5SuperClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "RecoEcal/EgammaClusterAlgos/interface/Multi5x5BremRecoveryClusterAlgo.h"

//


class Multi5x5SuperClusterProducer : public edm::EDProducer 
{
  
  public:

      Multi5x5SuperClusterProducer(const edm::ParameterSet& ps);

      ~Multi5x5SuperClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events
 
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

      Multi5x5BremRecoveryClusterAlgo * bremAlgo_p;

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

