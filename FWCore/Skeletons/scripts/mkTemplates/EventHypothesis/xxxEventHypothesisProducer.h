#ifndef EventHypothesis_xxxEventHypothesis_xxxEventHypothesisProducer_h
#define EventHypothesis_xxxEventHypothesis_xxxEventHypothesisProducer_h
#define xxxEventHypothesis_h
// -*- C++ -*-
//// -*- C++ -*-
//
// Package:    xxxEventHypothesis
// Class:      xxxEventHypothesis
//
/**


*/
//-------------------------------------------------------------------------------------
//!\class xxxEventHypothesisProducer xxxEventHypothesisProducer.cc skelsubys/xxxEventHypothesis/interface/xxxEventHypothesisProducer.h
//!\brief YOUR COMMENTS GO HERE
//!
//!
//! A long description of the event hypothesis producer class should go here.
//! 
//!
//-------------------------------------------------------------------------------------
//
//
// Original Author:  John Doe
//         Created:  day-mon-xx
// RCS(Id)
//


#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"


#include "skelsubsys/xxxEventHypothesis/interface/xxxEventHypothesis.h"


namespace pat {

  class xxxEventHypothesisProducer : public edm::EDProducer {

    public:

      explicit xxxEventHypothesisProducer(const edm::ParameterSet & iConfig);
      ~xxxEventHypothesisProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:

      // Here is a list of common includes.
      edm::InputTag      muonSrc_;
      edm::InputTag      electronSrc_;
      edm::InputTag      tauSrc_;
      edm::InputTag      photonSrc_;
      edm::InputTag      jetSrc_;
      edm::InputTag      metSrc_;
      // Here is the output tag name
      edm::OutputTag     outputName_;

  };


}

#endif
