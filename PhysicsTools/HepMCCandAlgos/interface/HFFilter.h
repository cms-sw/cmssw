#ifndef PhysicsTools_HepMCCandAlgos_interface_HFFilter_h
#define PhysicsTools_HepMCCandAlgos_interface_HFFilter_h

// -*- C++ -*-
//
// Package:    HFFilter
// Class:      HFFilter
// 
/**\class HFFilter HFFilter.cc PhysicsTools/HFFilter/src/HFFilter.cc

 Description: Filter to see if there are heavy flavor GenJets in this event

 Implementation:
     The implementation is simple, it loops over the GenJets and checks if any constituents
     have a pdg ID that matches a list. It also has a switch to count objects from a gluon parent,
     so the user can turn off counting gluon splitting. 
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Tue Apr  8 16:19:45 CDT 2008
// $Id: HFFilter.h,v 1.3 2013/02/27 23:16:51 wmtan Exp $
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class declaration
//

class HFFilter : public edm::EDFilter {
   public:
      explicit HFFilter(const edm::ParameterSet&);
      ~HFFilter();

      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

   private:      
      // ----------member data ---------------------------
      edm::InputTag    genJetsCollName_;        // Input GenJetsCollection
      double           ptMin_;                  // Min pt
      double           etaMax_;                 // Max abs(eta)
};

#endif
