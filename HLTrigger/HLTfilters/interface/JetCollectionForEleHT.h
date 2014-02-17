// -*- C++ -*-
//
// Package:    JetCollectionForEleHT
// Class:      JetCollectionForEleHT
// 
/**\class JetCollectionForEleHT JetCollectionForEleHT.cc HLTrigger/JetCollectionForEleHT/src/JetCollectionForEleHT.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Massimiliano Chiorboli,40 4-A01,+41227671535,
//         Created:  Mon Oct  4 11:57:35 CEST 2010
// $Id: JetCollectionForEleHT.h,v 1.2 2011/02/11 20:55:23 wdd Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "TVector3.h"


namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class JetCollectionForEleHT : public edm::EDProducer {
   public:
      explicit JetCollectionForEleHT(const edm::ParameterSet&);
      ~JetCollectionForEleHT();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


  edm::InputTag hltElectronTag;
  edm::InputTag sourceJetTag;
  
  float minDeltaR_; //min dR for jets and electrons not to match

      
      // ----------member data ---------------------------
};
