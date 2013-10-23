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
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaTowerIsolation;

//this class produces either Hcal isolation or H for H/E  depending if doEtSum=true or false
//H for H/E = towers behind SC, hcal isolation has these towers excluded
//a rho correction can be applied

class EgammaHLTBcHcalIsolationProducersRegional : public edm::EDProducer {
public:
  explicit EgammaHLTBcHcalIsolationProducersRegional(const edm::ParameterSet&);
  ~EgammaHLTBcHcalIsolationProducersRegional();

private:
  EgammaHLTBcHcalIsolationProducersRegional(const EgammaHLTBcHcalIsolationProducersRegional& rhs){}
  EgammaHLTBcHcalIsolationProducersRegional& operator=(const EgammaHLTBcHcalIsolationProducersRegional& rhs){ return *this; }
  
public:
  virtual void produce(edm::Event&, const edm::EventSetup&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  edm::EDGetTokenT<CaloTowerCollection> caloTowerProducer_;
  edm::EDGetTokenT<double> rhoProducer_;

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
  bool useSingleTower_;

  ElectronHcalHelper::Configuration hcalCfg_;
  ElectronHcalHelper *hcalHelper_;
};

