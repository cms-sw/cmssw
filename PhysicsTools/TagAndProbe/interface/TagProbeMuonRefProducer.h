#ifndef src_PhysicsTools_TagProbeMuonRefProducer_h
#define src_PhysicsTools_TagProbeMuonRefProducer_h
// -*- C++ -*-
//
// Package:     PhysicsTools
// Class  :     TagProbeMuonRefProducer
// 
/**\class TagProbeMuonRefProducer MuonRefProducer.h src/PhysicsTools/interface/MuonRefProducer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Valerie Halyo
//         Created:  Wed Oct  8 11:08:22 CDT 2008
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
#include "DataFormats/Common/interface/RefToBase.h" 
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


#include <string>
//
// class decleration
//

class TagProbeMuonRefProducer : public edm::EDProducer 
{
   public:
      explicit TagProbeMuonRefProducer(const edm::ParameterSet&);
      ~TagProbeMuonRefProducer();


      bool selectMuonIdAlgo(const reco::Muon& muonCand);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      

      // ----------member data ---------------------------
      
      edm::InputTag probeCollection_;

      bool   useCharge_;
      double ptMin_;
      int    charge_;
      int    nhits_;
      double nchi2_;
      double d0_;
      double z0_;
      double dz0_;
      std::string muonIdAlgo_;

      // ----------member data ---------------------------
};




#endif
