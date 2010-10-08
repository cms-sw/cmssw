#ifndef PhysicsTools_PatAlgos_PATTriggerEventProducer_h
#define PhysicsTools_PatAlgos_PATTriggerEventProducer_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerEventProducer
//
// $Id: PATTriggerEventProducer.h,v 1.8 2010/04/20 21:39:47 vadler Exp $
//
/**
  \class    pat::PATTriggerEventProducer PATTriggerEventProducer.h "PhysicsTools/PatAlgos/plugins/PATTriggerEventProducer.h"
  \brief    Produces the pat::TriggerEvent.

   [...]

  \author   Volker Adler
  \version  $Id: PATTriggerEventProducer.h,v 1.8 2010/04/20 21:39:47 vadler Exp $
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

      edm::ConditionsInRunBlock    condRun_;
      edm::ConditionsInLumiBlock   condLumi_;
      bool                         gtCondRunInit_;
      bool                         gtCondLumiInit_;
      HLTConfigProvider            hltConfig_;
      bool                         hltConfigInit_;
      std::string                  nameProcess_;        // configuration
      edm::InputTag                tagTriggerResults_;  // configuration (optional with default)
      edm::InputTag                tagTriggerProducer_; // configuration (optional with default)
      edm::InputTag                tagCondGt_;          // configuration (optional with default)
      edm::InputTag                tagL1Gt_;            // configuration (optional with default)
      std::vector< edm::InputTag > tagsTriggerMatcher_; // configuration (optional)

  };

}


#endif
