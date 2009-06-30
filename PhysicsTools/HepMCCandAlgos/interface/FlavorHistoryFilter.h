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
// $Id: FlavorHistoryFilter.h,v 1.6 2008/12/16 21:08:52 srappocc Exp $
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


#include "DataFormats/HepMCCandidate/interface/FlavorHistoryEvent.h"
#include "DataFormats/HepMCCandidate/interface/FlavorHistory.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

//
// class declaration
//

class FlavorHistoryFilter : public edm::EDFilter {
   public:
     typedef reco::FlavorHistory::FLAVOR_T flavor_type;
     typedef std::vector<int>              flavor_vector;

      explicit FlavorHistoryFilter(const edm::ParameterSet&);
      ~FlavorHistoryFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag   src_;            // Input flavor history collection name
      std::string     schemeName_;     // Which scheme to use
      int             flavor_;         // Flavor to examine
      int             noutput_;        // Required number of output HF jets
      flavor_vector   flavorSource_;   // which type to filter on
      double          minPt_;          // For pt scheme
      double          minDR_;          // For deltaR scheme
      double          maxDR_;          // For deltaR scheme
      bool            verbose_;        // verbosity

      
};


#endif 
