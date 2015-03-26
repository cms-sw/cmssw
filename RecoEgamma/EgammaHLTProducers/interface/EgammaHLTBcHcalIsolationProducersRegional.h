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
#include "FWCore/Framework/interface/stream/EDProducer.h"
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

class EgammaHLTBcHcalIsolationProducersRegional : public edm::stream::EDProducer<> {
public:
  explicit EgammaHLTBcHcalIsolationProducersRegional(const edm::ParameterSet&);
  ~EgammaHLTBcHcalIsolationProducersRegional();

  // non-copiable
  EgammaHLTBcHcalIsolationProducersRegional(EgammaHLTBcHcalIsolationProducersRegional const &) = delete;
  EgammaHLTBcHcalIsolationProducersRegional& operator=(EgammaHLTBcHcalIsolationProducersRegional const &) = delete;

public:
  virtual void produce(edm::Event&, const edm::EventSetup&) override final;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const bool  doRhoCorrection_;
  const float rhoScale_;
  const float rhoMax_;
  const bool  doEtSum_;
  const float etMin_;
  const float innerCone_;
  const float outerCone_;
  const int   depth_;
  const float effectiveAreaBarrel_;
  const float effectiveAreaEndcap_;
  const bool  useSingleTower_;

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<CaloTowerCollection>               caloTowerProducer_;
  const edm::EDGetTokenT<double>                            rhoProducer_;

  ElectronHcalHelper *hcalHelper_;
};

