// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTBcHcalIsolationProducersRegional
// 
// Original Author:  Matteo Sani (UCSD)
//         Created:  Thu Nov 24 11:38:00 CEST 2011
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"

class EgammaTowerIsolation;

class EgammaHLTBcHcalIsolationProducersRegional : public edm::EDProducer {
public:
  explicit EgammaHLTBcHcalIsolationProducersRegional(const edm::ParameterSet&);
  ~EgammaHLTBcHcalIsolationProducersRegional();
  
private:
  EgammaHLTBcHcalIsolationProducersRegional(const EgammaHLTBcHcalIsolationProducersRegional& rhs){}
  EgammaHLTBcHcalIsolationProducersRegional& operator=(const EgammaHLTBcHcalIsolationProducersRegional& rhs){ return *this; }
  
public:
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag caloTowerProducer_;
  edm::InputTag rhoProducer_;

  bool doRhoCorrection_;
  float rhoScale_;
  float rhoMax_;
  bool doEtSum_;
  float etMin_;
  float innerCone_;
  float outerCone_;
  int depth_;
  float effectiveAreaBarrel_;
  float effectiveAreaEndcap_;

  ElectronHcalHelper::Configuration hcalCfg;

  EgammaTowerIsolation *isoAlgo_;
  ElectronHcalHelper *hcalHelper;
};

