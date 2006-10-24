// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTEcalIsolationProducers
// 
/**\class EgammaHLTEcalIsolationProducers EgammaHLTEcalIsolationProducers.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTEcalIsolationProducers.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTEcalIsolationProducers.h,v 1.3 2006/10/24 14:12:56 monicava Exp $
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

class EgammaHLTEcalIsolationProducers : public edm::EDProducer {
   public:
      explicit EgammaHLTEcalIsolationProducers(const edm::ParameterSet&);
      ~EgammaHLTEcalIsolationProducers();


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

  EgammaHLTEcalIsolation* test_;


};

