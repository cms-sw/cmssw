#ifndef PhysicsTools_HepMCCandAlgos_interface_FlavorHistoryFilter_h
#define PhysicsTools_HepMCCandAlgos_interface_FlavorHistoryFilter_h


// -*- C++ -*-
//
// Package:    FlavorHistoryFilter
// Class:      FlavorHistoryFilter
// 
/**\class FlavorHistoryFilter FlavorHistoryFilter.cc PhysicsTools/FlavorHistoryFilter/src/FlavorHistoryFilter.cc

 Description: 

 This now filters events hierarchically. Previously this was done at the python configuration
 level, which was cumbersome for users to use. 

 Now, the hierarchy is:

 Create prioritized paths to separate HF composition samples.
 
 These are exclusive priorities, so sample "i" will not overlap with "i+1".
 Note that the "dr" values below correspond to the dr between the
 matched genjet, and the sister genjet. 

 1) W+bb with >= 2 jets from the ME (dr > 0.5)
 2) W+b or W+bb with 1 jet from the ME
 3) W+cc from the ME (dr > 0.5)
 4) W+c or W+cc with 1 jet from the ME
 5) W+bb with 1 jet from the parton shower (dr == 0.0)
 6) W+cc with 1 jet from the parton shower (dr == 0.0)

 These are the "trash bin" samples that we're throwing away:

 7) W+bb with >= 2 partons but 1 jet from the ME (dr == 0.0)
 8) W+cc with >= 2 partons but 1 jet from the ME (dr == 0.0)
 9) W+bb with >= 2 partons but 2 jets from the PS (dr > 0.5)
 10)W+cc with >= 2 partons but 2 jets from the PS (dr > 0.5)

 And here is the true "light flavor" sample:

 11) Veto of all the previous (W+ light jets)

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Sat Jun 28 00:41:21 CDT 2008
// $Id: FlavorHistoryFilter.h,v 1.10 2013/02/27 23:16:51 wmtan Exp $
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

#include "PhysicsTools/HepMCCandAlgos/interface/FlavorHistorySelectorUtil.h"

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
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag   bsrc_;           // Input b flavor history collection name
      edm::InputTag   csrc_;           // Input c flavor history collection name
      int             pathToSelect_;   // Select any of the following paths:
      double          dr_;             // dr with which to cut off the events
      // Note! The "b" and "c" here refer to the number of matched b and c genjets, respectively
      reco::FlavorHistorySelectorUtil * bb_me_;      // To select bb->2 events from matrix element... Path 1 
      reco::FlavorHistorySelectorUtil *  b_me_;      // To select  b->1 events from matrix element... Path 2
      reco::FlavorHistorySelectorUtil * cc_me_;      // To select cc->2 events from matrix element... Path 3
      reco::FlavorHistorySelectorUtil *  c_me_;      // To select  c->1 events from matrix element... Path 4
      reco::FlavorHistorySelectorUtil *  b_ps_;      // To select bb->2 events from parton shower ... Path 5
      reco::FlavorHistorySelectorUtil *  c_ps_;      // To select cc->2 events from parton shower ... Path 6
      reco::FlavorHistorySelectorUtil * bb_me_comp_; // To select bb->1 events from matrix element... Path 7
      reco::FlavorHistorySelectorUtil * cc_me_comp_; // To select cc->1 events from matrix element... Path 8
      reco::FlavorHistorySelectorUtil *  b_ps_comp_; // To select bb->2 events from parton shower ... Path 9
      reco::FlavorHistorySelectorUtil *  c_ps_comp_; // To select cc->1 events from parton shower ... Path 10
                                                     // The veto of all of these is               ... Path 11
};


#endif 
