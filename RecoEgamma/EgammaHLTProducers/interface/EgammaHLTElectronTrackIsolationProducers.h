// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTElectronTrackIsolationProducers
// 
/**\class EgammaHLTElectronTrackIsolationProducers EgammaHLTElectronTrackIsolationProducers.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTElectronTrackIsolationProducers.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//
// $Id: EgammaHLTElectronTrackIsolationProducers.h,v 1.2 2008/10/10 14:05:43 covarell Exp $
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

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTTrackIsolation.h"

//
// class declaration
//

class EgammaHLTElectronTrackIsolationProducers : public edm::EDProducer {
   public:
      explicit EgammaHLTElectronTrackIsolationProducers(const edm::ParameterSet&);
      ~EgammaHLTElectronTrackIsolationProducers();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag electronProducer_;
  edm::InputTag trackProducer_;

  edm::ParameterSet conf_;

  double egTrkIsoPtMin_; 
  double egTrkIsoConeSize_;
  double egTrkIsoZSpan_;   
  double egTrkIsoRSpan_;  
  double egTrkIsoVetoConeSize_;
  bool egCheckForOtherEleInCone_;

  EgammaHLTTrackIsolation* test_;
};

