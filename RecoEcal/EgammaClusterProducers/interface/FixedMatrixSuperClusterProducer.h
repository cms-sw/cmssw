#ifndef RecoEcal_EgammaClusterProducers_FixedMatrixSuperClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_FixedMatrixSuperClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "RecoEcal/EgammaClusterAlgos/interface/FixedMatrixBremRecoveryClusterAlgo.h"

//


class FixedMatrixSuperClusterProducer : public edm::EDProducer 
{
  
  public:

      FixedMatrixSuperClusterProducer(const edm::ParameterSet& ps);

      ~FixedMatrixSuperClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events
 
      FixedMatrixBremRecoveryClusterAlgo::VerbosityLevel verbosity;

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

      FixedMatrixBremRecoveryClusterAlgo * bremAlgo_p;

      double totalE;
      int noSuperClusters;

      
      void getClusterRefVector(edm::Event& evt, std::string clusterProducer_, std::string clusterCollection_, reco::BasicClusterRefVector *);
  
      void produceSuperclustersForECALPart(edm::Event& evt, 
					   std::string clusterProducer, 
					   std::string clusterCollection,
					   std::string superclusterColection);

      void outputValidationInfo(reco::SuperClusterCollection &superclusterCollection);
    
      bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0));}
};

#endif

