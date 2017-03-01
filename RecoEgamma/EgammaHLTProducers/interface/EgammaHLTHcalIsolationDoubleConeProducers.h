// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTHcalIsolationDoubleConeProducers
// 
/**\class EgammaHLTHcalIsolationDoubleConeProducers EgammaHLTHcalIsolationDoubleConeProducers.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTHcalIsolationDoubleConeProducers.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTHcalIsolationDoubleConeProducers.h,v 1.4 2006/10/24 15:25:53 monicava Exp $
//
//
// mostly identical to EgammaHLTHcalIsolationRegionalProducers, but produces excludes  
// Hcal energy in an exclusion cone around the eg candidate


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTHcalIsolationDoubleCone.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTHcalIsolationDoubleConeProducers : public edm::EDProducer {
public:
  explicit EgammaHLTHcalIsolationDoubleConeProducers(const edm::ParameterSet&);
  ~EgammaHLTHcalIsolationDoubleConeProducers();
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  edm::EDGetTokenT<HBHERecHitCollection> hbRecHitProducer_;
  edm::EDGetTokenT<HFRecHitCollection> hfRecHitProducer_;

  double egHcalIsoPtMin_;
  double egHcalIsoConeSize_;
  double egHcalExclusion_;

  edm::ParameterSet conf_;
  
  EgammaHLTHcalIsolationDoubleCone* test_;
};

