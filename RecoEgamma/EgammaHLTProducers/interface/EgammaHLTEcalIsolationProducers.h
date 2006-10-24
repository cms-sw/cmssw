// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTEcalIsolationProducers
// 
/**\class EgammaHLTEcalIsolationProducers EgammaHLTEcalIsolationProducers.cc RecoEgamma/EgammaHLTEcalIsolationProducers/interface/EgammaHLTEcalIsolationProducers.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Monica Vazquez Acosta
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTEcalIsolationProducers.h,v 1.1 2006/06/27 17:35:19 monicava Exp $
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


};

