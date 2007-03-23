// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTHcalIsolationProducersRegional
// 
/**\class EgammaHLTHcalIsolationProducersRegional EgammaHLTHcalIsolationProducersRegional.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTHcalIsolationProducersRegional.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTHcalIsolationProducersRegional.h,v 1.4 2006/10/24 15:25:53 monicava Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolation.h"

//
// class declaration
//

class EgammaHLTHcalIsolationProducersRegional : public edm::EDProducer {
   public:
      explicit EgammaHLTHcalIsolationProducersRegional(const edm::ParameterSet&);
      ~EgammaHLTHcalIsolationProducersRegional();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag hbRecHitProducer_;
  edm::InputTag hfRecHitProducer_;

  double egHcalIsoPtMin_;
  double egHcalIsoConeSize_;

  edm::ParameterSet conf_;

  EgammaHLTHcalIsolation* test_;

};

