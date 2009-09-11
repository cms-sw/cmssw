#ifndef src_PhysicsTools_MuTrkMatchedRecoChargedCandRefProducer_h
#define src_PhysicsTools_MuTrkMatchedRecoChargedCandRefProducer_h
// -*- C++ -*-
//
// Package:     PhysicsTools
// Class  :     MuTrkMatchedRecoChargedCandRefProducer
// 
/**\class MuTrkMatchedRecoChargedCandRefProducer MuTrkMatchedRecoChargedCandRefProducer.h src/PhysicsTools/interface/MuTrkMatchedRecoChargedCandRefProducer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Valerie Halyo, Nadia Adam
//         Created:  Wed Oct  8 11:08:22 CDT 2008
// $Id: MuTrkMatchedRecoChargedCandRefProducer.h,v 1.2 2009/01/21 21:01:59 neadam Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h" 
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h" 
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"


//
// class decleration
//

class MuTrkMatchedRecoChargedCandRefProducer : public edm::EDProducer 
{
   public:
      explicit MuTrkMatchedRecoChargedCandRefProducer(const edm::ParameterSet&);
      ~MuTrkMatchedRecoChargedCandRefProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      

      // ----------member data ---------------------------
      
      edm::InputTag muonCollection_;
      edm::InputTag trackCollection_;

      // ----------member data ---------------------------
};




#endif
