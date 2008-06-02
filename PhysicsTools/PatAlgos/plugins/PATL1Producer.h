#ifndef PhysicsTools_PatAlgos_PATL1Producer_h
#define PhysicsTools_PatAlgos_PATL1Producer_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      PATL1Producer
//
/**
  \class    pat::PATL1Producer PATL1Producer.h "PhysicsTools/PatAlgos/interface/PATL1Producer.h"
  \brief    Produces a CandidateCollection of L1 trigger objects.

   A CandidateCollection of "firing" L1 trigger objects from a given L1 trigger is produced from L1 information available in AOD.

  \author   Volker Adler
  \version  $Id: PATL1Producer.h,v 1.2.2.1 2008/04/08 09:42:12 vadler Exp $
*/
//
// $Id: PATL1Producer.h,v 1.2.2.1 2008/04/08 09:42:12 vadler Exp $
//


#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


namespace pat {


  class PATL1Producer : public edm::EDProducer {

    public:

      explicit PATL1Producer( const edm::ParameterSet & iConfig );
      ~PATL1Producer();

    private:

      virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup );
      
      edm::InputTag particleMaps_;
      std::string   triggerName_;
      std::string   objectType_;

  };


}


#endif
