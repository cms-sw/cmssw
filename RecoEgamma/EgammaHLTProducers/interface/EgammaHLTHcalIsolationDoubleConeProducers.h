// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTHcalIsolationDoubleConeProducers
// 
/**\class EgammaHLTHcalIsolationDoubleConeProducers EgammaHLTHcalIsolationDoubleConeProducers.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTHcalIsolationDoubleConeProducers.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTHcalIsolationDoubleConeProducers.h,v 1.1 2007/05/31 19:45:56 mpieri Exp $
//
//
// mostly identical to EgammaHLTHcalIsolationRegionalProducers, but produces excludes  
// Hcal energy in an exclusion cone around the eg candidate


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolationDoubleCone.h"

//
// class declaration
//

class EgammaHLTHcalIsolationDoubleConeProducers : public edm::EDProducer {
   public:
      explicit EgammaHLTHcalIsolationDoubleConeProducers(const edm::ParameterSet&);
      ~EgammaHLTHcalIsolationDoubleConeProducers();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag hbRecHitProducer_;
  edm::InputTag hfRecHitProducer_;

  double egHcalIsoPtMin_;
  double egHcalIsoConeSize_;
  double egHcalExclusion_;

  edm::ParameterSet conf_;
  
  EgammaHLTHcalIsolationDoubleCone* test_;

};

