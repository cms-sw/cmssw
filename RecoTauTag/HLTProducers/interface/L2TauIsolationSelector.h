/*
L2 Tau trigger Isolation Selector

Author: Michail Bachtis
University of Wisconsin-Madison
e-mail: bachtis@hep.wisc.edu
*/


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class L2TauIsolationSelector : public edm::EDProducer {
   public:
      explicit L2TauIsolationSelector(const edm::ParameterSet&);
      ~L2TauIsolationSelector();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

      edm::InputTag associationInput_;  
      
      //Create vars for Cuts
      double ECALIsolEt_;
      double TowerIsolEt_;
      double Cluster_etaRMS_;
      double Cluster_phiRMS_;
      double Cluster_drRMS_;
      int    Cluster_nClusters_;
      double JetEt_;
      double SeedTowerEt_;
    
};
