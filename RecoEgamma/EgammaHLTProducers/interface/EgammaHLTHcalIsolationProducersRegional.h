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
// $Id: EgammaHLTHcalIsolationProducersRegional.h,v 1.2 2010/08/12 15:17:28 sharper Exp $
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

class EgammaHLTHcalIsolation;

//
// class declaration
//

class EgammaHLTHcalIsolationProducersRegional : public edm::EDProducer {
public:
  explicit EgammaHLTHcalIsolationProducersRegional(const edm::ParameterSet&);
  ~EgammaHLTHcalIsolationProducersRegional();
  
  //now we need to disable (or even define) copy, assignment operators as we own a pointer
private:
  EgammaHLTHcalIsolationProducersRegional(const EgammaHLTHcalIsolationProducersRegional& rhs){}
  EgammaHLTHcalIsolationProducersRegional& operator=(const EgammaHLTHcalIsolationProducersRegional& rhs){return *this;}
  
public:
virtual void produce(edm::Event&, const edm::EventSetup&);

private:
      // ----------member data ---------------------------

  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag hbheRecHitProducer_;
  edm::InputTag rhoProducer_;

  bool doEtSum_;
  float effectiveAreaBarrel_;
  float effectiveAreaEndcap_;

  EgammaHLTHcalIsolation* isolAlgo_;
};

