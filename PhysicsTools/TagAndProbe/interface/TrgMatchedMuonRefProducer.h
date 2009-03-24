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
// $Id: TrgMatchedMuonRefProducer.h,v 1.2 2009/01/21 21:01:59 neadam Exp $
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
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      

      // ----------member data ---------------------------
      
      edm::InputTag probeCollection_;

      edm::InputTag triggerEventTag_;
      std::vector<edm::InputTag> muonFilterTags_;

      // Matching parameters
      double delRMatchingCut_;
      double delPtRelMatchingCut_;

      // Some details about the matching
      bool usePtMatching_;

      // ----------member data ---------------------------
};




#endif
