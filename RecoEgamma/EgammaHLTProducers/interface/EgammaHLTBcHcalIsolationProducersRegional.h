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
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"

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
  ~EgammaHLTBcHcalIsolationProducersRegional() override;

  // non-copiable
  EgammaHLTBcHcalIsolationProducersRegional(EgammaHLTBcHcalIsolationProducersRegional const &) = delete;
  EgammaHLTBcHcalIsolationProducersRegional& operator=(EgammaHLTBcHcalIsolationProducersRegional const &) = delete;

public:
  void produce(edm::Event&, const edm::EventSetup&) final;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const bool  doEtSum_;
  const double etMin_;
  const double innerCone_;
  const double outerCone_;
  const int   depth_;
  const bool  useSingleTower_;

  const bool  doRhoCorrection_;
  const double rhoScale_;
  const double rhoMax_;
  const std::vector<double> effectiveAreas_;
  const std::vector<double> absEtaLowEdges_;

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<CaloTowerCollection>               caloTowerProducer_;
  const edm::EDGetTokenT<double>                            rhoProducer_;

  ElectronHcalHelper *hcalHelper_;
};

