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

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class EgammaHLTR9Producer : public edm::EDProducer {
   public:
      explicit EgammaHLTR9Producer(const edm::ParameterSet&);
      ~EgammaHLTR9Producer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  edm::InputTag ecalRechitEBTag_;
  edm::InputTag ecalRechitEETag_;
  edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEBToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEEToken_;
  bool useSwissCross_;
  
  edm::ParameterSet conf_;

};

