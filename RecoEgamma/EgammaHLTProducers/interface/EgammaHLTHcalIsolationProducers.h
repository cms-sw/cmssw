// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTHcalIsolationProducers
// 
/**\class EgammaHLTHcalIsolationProducers EgammaHLTHcalIsolationProducers.cc RecoEgamma/EgammaHLTHcalIsolationProducers/interface/EgammaHLTHcalIsolationProducers.h
*/
//
// Original Author:  Monica Vazquez Acosta
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTHcalIsolationProducers.h,v 1.1 2006/06/27 17:35:36 monicava Exp $
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

//
// class declaration
//

class EgammaHLTHcalIsolationProducers : public edm::EDProducer {
   public:
      explicit EgammaHLTHcalIsolationProducers(const edm::ParameterSet&);
      ~EgammaHLTHcalIsolationProducers();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag hbRecHitProducer_;
  edm::InputTag hfRecHitProducer_;

  double egHcalIsoPtMin_;
  double egHcalIsoConeSize_;

  edm::ParameterSet conf_;


};

