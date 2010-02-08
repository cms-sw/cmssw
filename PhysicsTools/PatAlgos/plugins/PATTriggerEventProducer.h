#ifndef PhysicsTools_PatAlgos_PATTriggerEventProducer_h
#define PhysicsTools_PatAlgos_PATTriggerEventProducer_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerEventProducer
//
// $Id: PATTriggerEventProducer.h,v 1.4 2010/01/12 19:28:36 vadler Exp $
//
/**
  \class    pat::PATTriggerEventProducer PATTriggerEventProducer.h "PhysicsTools/PatAlgos/plugins/PATTriggerEventProducer.h"
  \brief    Produces the pat::TriggerEvent in PAT layer 1.

   [...]

  \author   Volker Adler
  \version  $Id: PATTriggerEventProducer.h,v 1.4 2010/01/12 19:28:36 vadler Exp $
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/PatCandidates/interface/TriggerObject.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/TriggerFilter.h"
#include "DataFormats/PatCandidates/interface/TriggerPath.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"


namespace pat {

  class PATTriggerEventProducer : public edm::EDProducer {

    public:

      explicit PATTriggerEventProducer( const edm::ParameterSet & iConfig );
      ~PATTriggerEventProducer();

    private:

      virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup );

      HLTConfigProvider            hltConfig_;
      std::string                  nameProcess_;
      edm::InputTag                tagTriggerResults_;
      edm::InputTag                tagTriggerProducer_;
      std::vector< edm::InputTag > tagsTriggerMatcher_;

  };

}


#endif
