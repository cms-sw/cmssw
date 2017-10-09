#ifndef RecoHi_HiEgammaAlgos_HiSuperClusterProducer_h_
#define RecoHi_HiEgammaAlgos_HiSuperClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "RecoHI/HiEgammaAlgos/interface/HiBremRecoveryClusterAlgo.h"

//


class HiSuperClusterProducer : public edm::stream::EDProducer<> 
{
  
  public:

      HiSuperClusterProducer(const edm::ParameterSet& ps);

      ~HiSuperClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob();

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events
 
      HiBremRecoveryClusterAlgo::VerbosityLevel verbosity;

      std::string endcapSuperclusterCollection_;
      std::string barrelSuperclusterCollection_;

      edm::EDGetTokenT<reco::BasicClusterCollection>  eeClustersToken_;
      edm::EDGetTokenT<reco::BasicClusterCollection>  ebClustersToken_;

      float barrelEtaSearchRoad_;
      float barrelPhiSearchRoad_;
      float endcapEtaSearchRoad_; 
      float endcapPhiSearchRoad_;
      float seedTransverseEnergyThreshold_;
      float barrelBCEnergyThreshold_;
      float endcapBCEnergyThreshold_;

      bool doBarrel_;
      bool doEndcaps_;

      HiBremRecoveryClusterAlgo * bremAlgo_p;

      double totalE;
      int noSuperClusters;

      
      void getClusterPtrVector(edm::Event& evt, const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken, reco::CaloClusterPtrVector *);
  
      void produceSuperclustersForECALPart(edm::Event& evt, 
					   const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken,
					   std::string superclusterColection);

      void outputValidationInfo(reco::SuperClusterCollection &superclusterCollection);
    
      bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0));}
};


#endif
