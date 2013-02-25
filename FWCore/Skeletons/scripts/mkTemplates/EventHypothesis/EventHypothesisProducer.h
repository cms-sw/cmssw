#ifndef EventHypothesis___class_____class__Producer_h
#define EventHypothesis___class_____class__Producer_h
#define __class___h
// -*- C++ -*-
//
// Package:    __pkgname__
// Class:      __class__
//
//-------------------------------------------------------------------------------------
//\class __class__Producer __class__Producer.cc __subsys__/__pkgname__/plugins/__class__Producer.h
//\brief YOUR COMMENTS GO HERE
//
//
// A long description of the event hypothesis producer class should go here.
// 
//
//-------------------------------------------------------------------------------------
//
//
// Original Author:  __author__
//         Created:  __date__
// __rcsid__
//


#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"


#include "__subsys__/__pkgname__/plugins/__class__.h"


namespace pat {

  class __class__Producer : public edm::EDProducer {

    public:

      explicit __class__Producer(const edm::ParameterSet & iConfig);
      ~__class__Producer();

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
