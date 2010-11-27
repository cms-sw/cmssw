#ifndef PhysicsTools_PatAlgos_PATTriggerEventProducer_h
#define PhysicsTools_PatAlgos_PATTriggerEventProducer_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerEventProducer
//
// $Id: PATTriggerEventProducer.h,v 1.9.2.1 2010/10/31 16:20:32 vadler Exp $
//
/**
  \class    pat::PATTriggerEventProducer PATTriggerEventProducer.h "PhysicsTools/PatAlgos/plugins/PATTriggerEventProducer.h"
  \brief    Produces the central entry point to full PAT trigger information

   This producer extract general trigger and conditions information from
   - the edm::TriggerResults written by the HLT process,
   - the ConditionsInEdm products,
   - the process history and
   - the GlobalTrigger information in the event and the event setup
   and writes it together with links to the full PAT trigger information collections and PAT trigger match results to
   - the pat::TriggerEvent

   For me information, s.
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger

  \author   Volker Adler
  \version  $Id: PATTriggerEventProducer.h,v 1.9.2.1 2010/10/31 16:20:32 vadler Exp $
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"


namespace pat {

  class PATTriggerEventProducer : public edm::EDProducer {

    public:

      explicit PATTriggerEventProducer( const edm::ParameterSet & iConfig );
      ~PATTriggerEventProducer() {};

    private:

      virtual void beginRun( edm::Run & iRun, const edm::EventSetup & iSetup );
      virtual void beginLuminosityBlock( edm::LuminosityBlock & iLumi, const edm::EventSetup & iSetup );
      virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup );

      std::string                  nameProcess_;        // configuration
      bool                         autoProcessName_;
      edm::InputTag                tagTriggerProducer_; // configuration (optional with default)
      std::vector< edm::InputTag > tagsTriggerMatcher_; // configuration (optional)
      // L1
      edm::InputTag                tagL1Gt_;            // configuration (optional with default)
      // HLT
      HLTConfigProvider            hltConfig_;
      bool                         hltConfigInit_;
      edm::InputTag                tagTriggerResults_;  // configuration (optional with default)
      edm::InputTag                tagTriggerEvent_;    // configuration (optional with default)
      // Conditions
      edm::InputTag                tagCondGt_;          // configuration (optional with default)
      edm::ConditionsInRunBlock    condRun_;
      edm::ConditionsInLumiBlock   condLumi_;
      bool                         gtCondRunInit_;
      bool                         gtCondLumiInit_;

  };

}


#endif
