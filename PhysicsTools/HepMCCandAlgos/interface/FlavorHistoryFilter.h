#ifndef PhysicsTools_HepMCCandAlgos_interface_FlavorHistoryFilter_h
#define PhysicsTools_HepMCCandAlgos_interface_FlavorHistoryFilter_h


// -*- C++ -*-
//
// Package:    FlavorHistoryFilter
// Class:      FlavorHistoryFilter
// 
/**\class FlavorHistoryFilter FlavorHistoryFilter.cc PhysicsTools/FlavorHistoryFilter/src/FlavorHistoryFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Sat Jun 28 00:41:21 CDT 2008
// $Id: FlavorHistoryFilter.h,v 1.2 2008/07/29 17:36:05 srappocc Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/HepMCCandidate/interface/FlavorHistory.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

//
// class declaration
//

class FlavorHistoryFilter : public edm::EDFilter {
   public:
      explicit FlavorHistoryFilter(const edm::ParameterSet&);
      ~FlavorHistoryFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag   src_;            // Input flavor history collection name
      edm::InputTag   jets_;           // Jet collection to match to
      int             type_;           // Returns "true" only if FlavorHistory::flavorSource == type
      int             nmatched_;       // Returns "true" only if number of partons matched == nmatched
      double          matchDR_;        // Delta R between gen partons and gen jets
      double          minPt_;          // For pt scheme
      double          minDR_;          // For deltaR scheme
      double          maxDR_;          // For deltaR scheme
      std::string     scheme_;         // Which scheme to use
      bool            requireSisters_; // Require sisters to exist to pass
      bool            verbose_;        // verbosity

      reco::GenJetCollection::const_iterator
	getClosestJet( edm::Handle<reco::GenJetCollection> const & pJets,
		       reco::CandidatePtr const & parton ) const;
      
};


#endif 
