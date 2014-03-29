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

#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"

class L2TauRelaxingIsolationSelector : public edm::EDProducer {
   public:
      explicit L2TauRelaxingIsolationSelector(const edm::ParameterSet&);
      ~L2TauRelaxingIsolationSelector();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
      

      //Association class Input
      edm::EDGetTokenT<reco::L2TauInfoAssociation> associationInput_;  
      
      //Sliding Cuts
      std::vector<double> ecalIsolEt_;
      std::vector<double> towerIsolEt_;
      std::vector<double> nClusters_;
      std::vector<double> phiRMS_;
      std::vector<double> etaRMS_;
      std::vector<double> drRMS_;

      //Cuts of the Style This > Something
      double et_;
      double seedTowerEt_;
    
};
