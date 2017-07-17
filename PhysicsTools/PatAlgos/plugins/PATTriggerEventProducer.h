#ifndef PhysicsTools_PatAlgos_PATTriggerEventProducer_h
#define PhysicsTools_PatAlgos_PATTriggerEventProducer_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerEventProducer
//
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
  \version  $Id: PATTriggerEventProducer.h,v 1.11 2010/11/27 15:16:20 vadler Exp $
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"

#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"


namespace pat {

  class PATTriggerEventProducer : public edm::stream::EDProducer<> {

    public:

      explicit PATTriggerEventProducer( const edm::ParameterSet & iConfig );
      ~PATTriggerEventProducer() {};

    private:

      virtual void beginRun(const edm::Run & iRun, const edm::EventSetup& iSetup) override;
      virtual void beginLuminosityBlock(const edm::LuminosityBlock & iLumi, const edm::EventSetup& iSetup) override;
      virtual void produce( edm::Event & iEvent, const edm::EventSetup& iSetup) override;

      std::string                  nameProcess_;        // configuration
      bool                         autoProcessName_;
      edm::InputTag                tagTriggerProducer_; // configuration (optional with default)
      edm::EDGetTokenT< TriggerAlgorithmCollection > triggerAlgorithmCollectionToken_;
      edm::EDGetTokenT< TriggerConditionCollection > triggerConditionCollectionToken_;
      edm::EDGetTokenT< TriggerPathCollection >      triggerPathCollectionToken_;
      edm::EDGetTokenT< TriggerFilterCollection >    triggerFilterCollectionToken_;
      edm::EDGetTokenT< TriggerObjectCollection >    triggerObjectCollectionToken_;
      std::vector< edm::InputTag > tagsTriggerMatcher_; // configuration (optional)
      std::vector< edm::EDGetTokenT< TriggerObjectStandAloneMatch > > triggerMatcherTokens_;
      // L1
      edm::InputTag                tagL1Gt_;            // configuration (optional with default)
      edm::EDGetTokenT< L1GlobalTriggerReadoutRecord > l1GtToken_;
      // HLT
      HLTConfigProvider            hltConfig_;
      bool                         hltConfigInit_;
      edm::InputTag                tagTriggerResults_;  // configuration (optional with default)
      edm::GetterOfProducts< edm::TriggerResults > triggerResultsGetter_;
      edm::InputTag                tagTriggerEvent_;    // configuration (optional with default)
      // Conditions
      edm::InputTag                tagCondGt_;          // configuration (optional with default)
      edm::EDGetTokenT< edm::ConditionsInRunBlock > tagCondGtRunToken_;
      edm::EDGetTokenT< edm::ConditionsInLumiBlock > tagCondGtLumiToken_;
      edm::EDGetTokenT< edm::ConditionsInEventBlock > tagCondGtEventToken_;
      edm::ConditionsInRunBlock    condRun_;
      edm::ConditionsInLumiBlock   condLumi_;
      bool                         gtCondRunInit_;
      bool                         gtCondLumiInit_;

  };

}


#endif
