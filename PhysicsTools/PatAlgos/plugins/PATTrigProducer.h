#ifndef PhysicsTools_PatAlgos_PATTrigProducer_h
#define PhysicsTools_PatAlgos_PATTrigProducer_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      PATTrigProducer
//
/**
  \class    pat::PATTrigProducer PATTrigProducer.h "PhysicsTools/PatAlgos/plugins/PATTrigProducer.h"
  \brief    Produces a CandidateCollection of trigger objects.

   A CandidateCollection of trigger objects from a given filter is produced from trigger information available in AOD.

  \author   Volker Adler
  \version  $Id$
*/
//
// $Id$
//


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


namespace pat {


  class PATTrigProducer : public edm::EDProducer {

    public:

      explicit PATTrigProducer( const edm::ParameterSet & iConfig );
      ~PATTrigProducer();

    private:

      virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup );
      
      edm::InputTag triggerEvent_;
      edm::InputTag filterName_;

  };


}


#endif
