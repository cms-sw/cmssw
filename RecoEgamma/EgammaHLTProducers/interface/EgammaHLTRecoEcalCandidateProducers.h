// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTRecoEcalCandidateProducers
// 
/**\class EgammaHLTRecoEcalCandidateProducers.h EgammaHLTRecoEcalCandidateProducers.cc  RecoEgamma/EgammaHLTProducers/interface/EgammaHLTRecoEcalCandidateProducers.h.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id:
//
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"



// EgammaHLTRecoEcalCandidateProducers inherits from EDProducer, so it can be a module:
class EgammaHLTRecoEcalCandidateProducers : public edm::EDProducer {

 public:

  EgammaHLTRecoEcalCandidateProducers (const edm::ParameterSet& ps);
  ~EgammaHLTRecoEcalCandidateProducers();


  virtual void beginJob(void);
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:
  std::string recoEcalCandidateCollection_;
  edm::InputTag scHybridBarrelProducer_;
  edm::InputTag scIslandEndcapProducer_;
  edm::ParameterSet conf_;
};


