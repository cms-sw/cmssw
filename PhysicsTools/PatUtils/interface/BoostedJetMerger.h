// -*- C++ -*-
//
// Package:    BoostedJetMerger
// Class:      BoostedJetMerger
// 
// \class BoostedJetMerger BoostedJetMerger.h PhysicsTools/PatUtils/interface/BoostedJetMerger.h
// Description: Class to "deswizzle" information from various pat::Jet collections.
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Thu May  1 11:37:48 CDT 2008
// $Id: BoostedJetMerger.cc,v 1.1 2013/03/07 20:13:55 srappocc Exp $
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
#include "DataFormats/PatCandidates/interface/Jet.h"

//
// class decleration
//


/// Predicate to use for find_if.
/// This checks whether a given edm::Ptr<reco::Candidate>
/// (as you would get from the reco::BasicJet daughters)
/// to see if it matches the original object ref of
/// another pat::Jet (which is to find the corrected / btagged
/// pat::Jet that corresponds to the subjet in question). 
struct FindCorrectedSubjet {
  // Input the daughter you're interested in checking
  FindCorrectedSubjet( edm::Ptr<reco::Candidate> const & da ) : 
    da_(da) {}

  // Predicate operator to compare an input pat::Jet to. 
  bool operator()( pat::Jet const & subjet ) const {
    edm::Ptr<reco::Candidate> subjetOrigRef = subjet.originalObjectRef();
    if ( da_ == subjetOrigRef ) {
      return true;
    }
    else return false;
  }

  edm::Ptr<reco::Candidate> da_;
};

class BoostedJetMerger : public edm::EDProducer {
   public:
      explicit BoostedJetMerger(const edm::ParameterSet&);
      ~BoostedJetMerger();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

      // data labels
      edm::EDGetTokenT<edm::View<pat::Jet> >  jetToken_;
      edm::EDGetTokenT<edm::View<pat::Jet> >  subjetToken_;
};
