// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTEcalIsolationProducersRegional
// 
/**\class EgammaHLTEcalIsolationProducersRegional EgammaHLTEcalIsolationProducersRegional.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTEcalIsolationProducersRegional.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTEcalIsolationProducersRegional.h,v 1.3 2011/12/19 11:16:45 sani Exp $
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
#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTEcalIsolation.h"


//
// class declaration
//

class EgammaHLTEcalIsolationProducersRegional : public edm::EDProducer {
   public:
      explicit EgammaHLTEcalIsolationProducersRegional(const edm::ParameterSet&);
      ~EgammaHLTEcalIsolationProducersRegional();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
  // ----------member data ---------------------------

  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag bcBarrelProducer_;
  edm::InputTag bcEndcapProducer_;
  edm::InputTag scIslandBarrelProducer_;
  edm::InputTag scIslandEndcapProducer_;

  edm::ParameterSet conf_;

  double  egEcalIsoEtMin_;
  double  egEcalIsoConeSize_;
  int algoType_;
  EgammaHLTEcalIsolation* test_;
};

