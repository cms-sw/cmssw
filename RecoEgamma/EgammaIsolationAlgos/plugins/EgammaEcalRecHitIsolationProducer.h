#ifndef EgammaIsolationProducers_EgammaEcalRecHitIsolationProducer_h
#define EgammaIsolationProducers_EgammaEcalRecHitIsolationProducer_h

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
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"

//
// class declaration
//

class EgammaEcalRecHitIsolationProducer : public edm::stream::EDProducer<> {
   public:
      explicit EgammaEcalRecHitIsolationProducer(const edm::ParameterSet&);
      ~EgammaEcalRecHitIsolationProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag emObjectProducer_;
  edm::InputTag ecalBarrelRecHitProducer_;
  edm::InputTag ecalBarrelRecHitCollection_;  
  edm::InputTag ecalEndcapRecHitProducer_;
  edm::InputTag ecalEndcapRecHitCollection_;
 
  double egIsoPtMinBarrel_; //minimum Et noise cut
  double egIsoEMinBarrel_;  //minimum E noise cut
  double egIsoPtMinEndcap_; //minimum Et noise cut
  double egIsoEMinEndcap_;  //minimum E noise cut
  double egIsoConeSizeOut_; //outer cone size
  double egIsoConeSizeInBarrel_; //inner cone size
  double egIsoConeSizeInEndcap_; //inner cone size
  double egIsoJurassicWidth_ ; // exclusion strip width for jurassic veto

  bool useIsolEt_; //switch for isolEt rather than isolE
  bool tryBoth_ ; // use rechits from barrel + endcap 
  bool subtract_ ; // subtract SC energy (allows veto cone of zero size)

  bool useNumCrystals_ ; // veto on number of crystals
  bool vetoClustered_ ;  // veto all clusterd rechits

  edm::ParameterSet conf_;

};

#endif
