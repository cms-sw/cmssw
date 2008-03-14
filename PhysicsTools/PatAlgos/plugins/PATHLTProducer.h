#ifndef PhysicsTools_PatAlgos_PATHLTProducer_h
#define PhysicsTools_PatAlgos_PATHLTProducer_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      PATHLTProducer
//
/**
  \class    pat::PATHLTProducer PATHLTProducer.h "PhysicsTools/PatAlgos/interface/PATHLTProducer.h"
  \brief    Produces a CandidateCollection of trigger objects.

   A CandidateCollection of "firing" trigger objects from a given filter in a given trigger path is produced from trigger information available in AOD.

  \author   Volker Adler
  \version  $Id: PATHLTProducer.h,v 1.1 2008/03/06 09:23:10 llista Exp $
*/
//
// $Id: PATHLTProducer.h,v 1.1 2008/03/06 09:23:10 llista Exp $
// based on the HLTCandProducer by florian.bechtel@cern.ch
//


#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


namespace pat {


  class PATHLTProducer : public edm::EDProducer {

    public:

      explicit PATHLTProducer(const edm::ParameterSet & iConfig);
      ~PATHLTProducer();

    private:

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      
      edm::InputTag triggerResults_;
      std::string   triggerName_;
      edm::InputTag filterName_;

  };


}


#endif
