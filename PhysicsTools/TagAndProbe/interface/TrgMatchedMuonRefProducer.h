#ifndef src_PhysicsTools_TrgMatchedMuonRefProducer_h
#define src_PhysicsTools_TrgMatchedMuonRefProducer_h
// -*- C++ -*-
//
// Package:     PhysicsTools
// Class  :     TrgMatchedMuonRefProducer
// 
/**\class TrgMatchedMuonRefProducer TrgMatchedMuonRefProducer.h src/PhysicsTools/interface/TrgMatchedMuonRefProducer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Wed Oct  8 11:08:22 CDT 2008
// $Id$
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h" 
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h" 
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h" 
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"


//
// class decleration
//

class TrgMatchedMuonRefProducer : public edm::EDProducer 
{
   public:
      explicit TrgMatchedMuonRefProducer(const edm::ParameterSet&);
      ~TrgMatchedMuonRefProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      

      bool MatchObjects( const trigger::TriggerObject& hltObj, 
			 const reco::CandidateBaseRef& tagObj,
			 bool exact = true );
      
      
      // ----------member data ---------------------------
      
      edm::InputTag probeCollection_;

      edm::InputTag triggerEventTag_;
      edm::InputTag hltL1Tag_;
      edm::InputTag hltTag_;


      // Matching parameters
      double delRMatchingCut_;
      double delPtRelMatchingCut_;


      // Some details about the matching
      bool simpleMatching_;
      bool doL1Matching_;
      bool usePtMatching_;

      // ----------member data ---------------------------
};




#endif
