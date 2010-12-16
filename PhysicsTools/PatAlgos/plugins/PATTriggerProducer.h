#ifndef PhysicsTools_PatAlgos_PATTriggerProducer_h
#define PhysicsTools_PatAlgos_PATTriggerProducer_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerProducer
//
// $Id: PATTriggerProducer.h,v 1.12.2.1 2010/10/31 16:20:32 vadler Exp $
//
/**
  \class    pat::PATTriggerProducer PATTriggerProducer.h "PhysicsTools/PatAlgos/plugins/PATTriggerProducer.h"
  \brief    Produces the pat::TriggerPathCollection, pat::TriggerFilterCollection and pat::TriggerObjectCollection.

   [...]

  \author   Volker Adler
  \version  $Id: PATTriggerProducer.h,v 1.12.2.1 2010/10/31 16:20:32 vadler Exp $
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"


namespace pat {

  class PATTriggerProducer : public edm::EDProducer {

    public:

      explicit PATTriggerProducer( const edm::ParameterSet & iConfig );
      ~PATTriggerProducer() {};

    private:

      virtual void beginRun( edm::Run & iRun, const edm::EventSetup & iSetup );
      virtual void beginLuminosityBlock( edm::LuminosityBlock & iLuminosityBlock, const edm::EventSetup & iSetup );
      virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup );

      // I need to make a copy of these two methods taking const Run and const LumiBlock
      // so that I can call them from 'produce', when auto-discovering process name
      void beginConstRun( const edm::Run & iRun, const edm::EventSetup & iSetup );
      void beginConstLuminosityBlock( const edm::LuminosityBlock & iLuminosityBlock, const edm::EventSetup & iSetup );

      std::string nameProcess_;     // configuration
      bool        autoProcessName_;
      bool        onlyStandAlone_;  // configuration
      // L1
      L1GtUtils     l1GtUtils_;
      bool          addL1Algos_;        // configuration (optional with default)
      edm::InputTag tagL1ExtraMu_;      // configuration (optional)
      edm::InputTag tagL1ExtraNoIsoEG_; // configuration (optional)
      edm::InputTag tagL1ExtraIsoEG_;   // configuration (optional)
      edm::InputTag tagL1ExtraCenJet_;  // configuration (optional)
      edm::InputTag tagL1ExtraForJet_;  // configuration (optional)
      edm::InputTag tagL1ExtraTauJet_;  // configuration (optional)
      edm::InputTag tagL1ExtraETM_;     // configuration (optional)
      edm::InputTag tagL1ExtraHTM_;     // configuration (optional)
      bool          autoProcessNameL1ExtraMu_;
      bool          autoProcessNameL1ExtraNoIsoEG_;
      bool          autoProcessNameL1ExtraIsoEG_;
      bool          autoProcessNameL1ExtraCenJet_;
      bool          autoProcessNameL1ExtraForJet_;
      bool          autoProcessNameL1ExtraTauJet_;
      bool          autoProcessNameL1ExtraETM_;
      bool          autoProcessNameL1ExtraHTM_;
      bool          saveL1Refs_;        // configuration (optional with default)
      // HLT
      HLTConfigProvider         hltConfig_;
      bool                      hltConfigInit_;
      edm::InputTag             tagTriggerResults_;     // configuration (optional with default)
      edm::InputTag             tagTriggerEvent_;       // configuration (optional with default)
      std::string               hltPrescaleLabel_;      // configuration (optional)
      std::string               labelHltPrescaleTable_; // configuration (optional)
      trigger::HLTPrescaleTable hltPrescaleTableRun_;
      trigger::HLTPrescaleTable hltPrescaleTableLumi_;
      bool                      addPathModuleLabels_;   // configuration (optional with default)

  };

}


#endif
