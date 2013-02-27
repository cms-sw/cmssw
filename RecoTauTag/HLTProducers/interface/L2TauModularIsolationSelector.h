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


class L2TauModularIsolationSelector : public edm::EDProducer {
   public:
      explicit L2TauModularIsolationSelector(const edm::ParameterSet&);
      ~L2TauModularIsolationSelector();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
      

      //Association class Input
      edm::InputTag associationInput_;  
      
      //Sliding Cuts (ECAL)

      std::vector<double> ecalIsolEt_;

      std::vector<double> nEcalClusters_;
      std::vector<double> ecalClusterPhiRMS_;
      std::vector<double> ecalClusterEtaRMS_;
      std::vector<double> ecalClusterDrRMS_;

      //Sliding Cuts (HCAL)

      std::vector<double> hcalIsolEt_;

      std::vector<double> nHcalClusters_;
      std::vector<double> hcalClusterPhiRMS_;
      std::vector<double> hcalClusterEtaRMS_;
      std::vector<double> hcalClusterDrRMS_;


      //Cuts of the Style This > Something
      double et_;
      double seedTowerEt_;
    
};
