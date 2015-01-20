#ifndef EgammaHLTProducers_EgammaHLTEcalRecIsolationProducer_h
#define EgammaHLTProducers_EgammaHLTEcalRecIsolationProducer_h

//*****************************************************************************
// File:      EgammaRecHitIsolationProducer.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, adapted from EgammaHcalIsolationProducer by S. Harper
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************

// -*- C++ -*-
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTEcalRecIsolationProducer : public edm::EDProducer {
 public:
  explicit EgammaHLTEcalRecIsolationProducer(const edm::ParameterSet&);
  ~EgammaHLTEcalRecIsolationProducer();
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  edm::EDGetTokenT<EcalRecHitCollection> ecalBarrelRecHitProducer_;
  edm::EDGetTokenT<EcalRecHitCollection> ecalEndcapRecHitProducer_;
  edm::EDGetTokenT<double> rhoProducer_;

  double egIsoPtMinBarrel_; //minimum Et noise cut
  double egIsoEMinBarrel_;  //minimum E noise cut
  double egIsoPtMinEndcap_; //minimum Et noise cut
  double egIsoEMinEndcap_;  //minimum E noise cut
  double egIsoConeSizeOut_; //outer cone size
  double egIsoConeSizeInBarrel_; //inner cone size
  double egIsoConeSizeInEndcap_; //inner cone size
  double egIsoJurassicWidth_ ; // exclusion strip width for jurassic veto
  float effectiveAreaBarrel_;
  float effectiveAreaEndcap_;

  bool doRhoCorrection_;
  float rhoScale_;
  float rhoMax_;

  bool useIsolEt_; //switch for isolEt rather than isolE
  bool tryBoth_ ; // use rechits from barrel + endcap
  bool subtract_ ; // subtract SC energy (allows veto cone of zero size)
  bool useNumCrystals_;// veto cones are specified in number of crystals not eta

  edm::ParameterSet conf_;
};
#endif
