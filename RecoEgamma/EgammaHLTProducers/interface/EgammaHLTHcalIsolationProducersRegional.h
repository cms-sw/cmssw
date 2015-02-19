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
#include "FWCore/Framework/interface/EDProducer.h"
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

class EgammaHLTHcalIsolationProducersRegional : public edm::EDProducer {
public:
  explicit EgammaHLTHcalIsolationProducersRegional(const edm::ParameterSet&);
  ~EgammaHLTHcalIsolationProducersRegional();

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  EgammaHLTHcalIsolationProducersRegional(const EgammaHLTHcalIsolationProducersRegional& rhs){}
  EgammaHLTHcalIsolationProducersRegional& operator=(const EgammaHLTHcalIsolationProducersRegional& rhs){return *this;}
  
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  edm::EDGetTokenT<HBHERecHitCollection> hbheRecHitProducer_;
  edm::EDGetTokenT<double> rhoProducer_;

  bool doRhoCorrection_;
  float rhoScale_;
  float rhoMax_;
  bool doEtSum_;
  float effectiveAreaBarrel_;
  float effectiveAreaEndcap_;

  EgammaHLTHcalIsolation* isolAlgo_;
};

