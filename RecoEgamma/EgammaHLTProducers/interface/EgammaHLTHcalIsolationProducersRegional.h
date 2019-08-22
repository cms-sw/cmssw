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
// $Id: EgammaHLTHcalIsolationProducersRegional.h,v 1.3 2011/12/19 11:17:28 sani Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTHcalIsolation;

class EgammaHLTHcalIsolationProducersRegional : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTHcalIsolationProducersRegional(const edm::ParameterSet&);
  ~EgammaHLTHcalIsolationProducersRegional() override;

  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<HBHERecHitCollection> hbheRecHitProducer_;
  const edm::EDGetTokenT<double> rhoProducer_;

  const bool doRhoCorrection_;
  const float rhoMax_;
  const float rhoScale_;
  const bool doEtSum_;
  const float effectiveAreaBarrel_;
  const float effectiveAreaEndcap_;

  EgammaHLTHcalIsolation const* const isolAlgo_;
};
