// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTR9Producer
// 
/**\class EgammaHLTR9Producer EgammaHLTR9Producer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTR9Producer.h
*/
//
// Original Author:  Roberto Covarelli (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTR9Producer.h,v 1.1 2010/02/11 10:06:38 ghezzi Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTR9Producer : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTR9Producer(const edm::ParameterSet&);
  ~EgammaHLTR9Producer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;

private:
      // ----------member data ---------------------------

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEBToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEEToken_;
  
  const bool useSwissCross_;
};

