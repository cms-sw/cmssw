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
  \version  $Id: PATHLTProducer.h,v 1.2 2008/03/05 14:56:50 fronga Exp $
*/
//
// $Id: PATHLTProducer.h,v 1.2 2008/03/05 14:56:50 fronga Exp $
// based on the HLTCandProducer by florian.bechtel@cern.ch
//


#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


using namespace std;
using namespace edm;

namespace pat {

  class PATHLTProducer : public EDProducer {

    public:

      explicit PATHLTProducer( const ParameterSet& iConfig );
      ~PATHLTProducer();

    private:

      virtual void produce( Event& iEvent, const EventSetup& iSetup );
      
      InputTag triggerResults_;
      string   triggerName_;
      InputTag filterName_;

  };


}


#endif
