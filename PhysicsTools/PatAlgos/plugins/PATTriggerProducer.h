#ifndef PhysicsTools_PatAlgos_PATTriggerProducer_h
#define PhysicsTools_PatAlgos_PATTriggerProducer_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerProducer
//
// $Id: PATTriggerProducer.h,v 1.1.2.2 2009/03/15 12:24:15 vadler Exp $
//
/**
  \class    pat::PATTriggerProducer PATTriggerProducer.h "PhysicsTools/PatAlgos/plugins/PATTriggerProducer.h"
  \brief    Produces the pat::TriggerPathCollection, pat::TriggerFilterCollection and pat::TriggerObjectCollection in PAT layer 0.

   [...]

  \author   Volker Adler
  \version  $Id: PATTriggerProducer.h,v 1.1.2.2 2009/03/15 12:24:15 vadler Exp $
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include <string>
#include <vector>
#include <map>
#include <cassert>

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/PatCandidates/interface/TriggerPath.h"
#include "DataFormats/PatCandidates/interface/TriggerFilter.h"
#include "DataFormats/PatCandidates/interface/TriggerObject.h"


namespace pat {

  class PATTriggerProducer : public edm::EDProducer {

    public:

      explicit PATTriggerProducer( const edm::ParameterSet & iConfig );
      ~PATTriggerProducer();

    private:

      virtual void beginRun( edm::Run & iRun, const edm::EventSetup & iSetup );
      virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup );
      
      HLTConfigProvider hltConfig_;
      std::string       nameProcess_;
      edm::InputTag     tagTriggerResults_;
      edm::InputTag     tagTriggerEvent_;
      
      // trigger path
      bool addPathModuleLabels_;

  };

}


#endif
