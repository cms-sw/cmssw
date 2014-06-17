#ifndef EgammaIsolationProducers_EgammaTowerIsolationProducer_h
#define EgammaIsolationProducers_EgammaTowerIsolationProducer_h

//*****************************************************************************
// File:      EgammaTowerIsolationProducer.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

// -*- C++ -*-
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

//
// class declaration
//

class EgammaTowerIsolationProducer : public edm::stream::EDProducer<> {
   public:
      explicit EgammaTowerIsolationProducer(const edm::ParameterSet&);
      ~EgammaTowerIsolationProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag emObjectProducer_;
  edm::InputTag towerProducer_;

  double egHcalIsoPtMin_;
  double egHcalIsoConeSizeOut_;
  double egHcalIsoConeSizeIn_;
  signed int egHcalDepth_;

  edm::ParameterSet conf_;

};

#endif
