// -*- C++ -*-
//
// Package:   BeamSplash
// Class:     BeamSPlash
//
//
// Original Author:  Luca Malgeri

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "DPGAnalysis/Skims/interface/BeamSplash.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

using namespace edm;
using namespace std;

BeamSplash::BeamSplash(const edm::ParameterSet& iConfig) {
  EBRecHitCollection_ = iConfig.getParameter<edm::InputTag>("ebrechitcollection");
  EERecHitCollection_ = iConfig.getParameter<edm::InputTag>("eerechitcollection");
  HBHERecHitCollection_ = iConfig.getParameter<edm::InputTag>("hbherechitcollection");

  EnergyCutTot = iConfig.getUntrackedParameter<double>("energycuttot");
  EnergyCutEcal = iConfig.getUntrackedParameter<double>("energycutecal");
  EnergyCutHcal = iConfig.getUntrackedParameter<double>("energycuthcal");
  applyfilter = iConfig.getUntrackedParameter<bool>("applyfilter", true);
}

BeamSplash::~BeamSplash() {}

bool BeamSplash::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool accepted = false;

  bool acceptedtot = false;
  bool acceptedEcal = false;
  bool acceptedHcal = false;

  int ievt = iEvent.id().event();
  int irun = iEvent.id().run();
  int ils = iEvent.luminosityBlock();
  int ibx = iEvent.bunchCrossing();

  double totene = 0;
  double ecalene = 0;
  double hcalene = 0;

  Handle<EBRecHitCollection> pEBRecHits;
  Handle<EERecHitCollection> pEERecHits;
  Handle<HBHERecHitCollection> pHBHERecHits;

  const EBRecHitCollection* EBRecHits = nullptr;
  const EERecHitCollection* EERecHits = nullptr;
  const HBHERecHitCollection* HBHERecHits = nullptr;

  if (!EBRecHitCollection_.label().empty() && !EBRecHitCollection_.instance().empty()) {
    iEvent.getByLabel(EBRecHitCollection_, pEBRecHits);
    if (pEBRecHits.isValid()) {
      EBRecHits = pEBRecHits.product();  // get a ptr to the product
    } else {
      edm::LogError("EcalRecHitError") << "Error! can't get the product " << EBRecHitCollection_.label();
    }
  }

  if (!EERecHitCollection_.label().empty() && !EERecHitCollection_.instance().empty()) {
    iEvent.getByLabel(EERecHitCollection_, pEERecHits);

    if (pEERecHits.isValid()) {
      EERecHits = pEERecHits.product();  // get a ptr to the product
    } else {
      edm::LogError("EcalRecHitError") << "Error! can't get the product " << EERecHitCollection_.label();
    }
  }

  if (!HBHERecHitCollection_.label().empty()) {
    iEvent.getByLabel(HBHERecHitCollection_, pHBHERecHits);

    if (pHBHERecHits.isValid()) {
      HBHERecHits = pHBHERecHits.product();  // get a ptr to the product
    } else {
      edm::LogError("HcalRecHitError") << "Error! can't get the product " << HBHERecHitCollection_.label();
    }
  }

  // now sum over them
  if (EBRecHits) {
    for (EBRecHitCollection::const_iterator it = EBRecHits->begin(); it != EBRecHits->end(); ++it) {
      totene += it->energy();
      ecalene += it->energy();
    }
  }
  if (EERecHits) {
    for (EERecHitCollection::const_iterator it = EERecHits->begin(); it != EERecHits->end(); ++it) {
      totene += it->energy();
      ecalene += it->energy();
    }
  }
  if (HBHERecHits) {
    for (HBHERecHitCollection::const_iterator it = HBHERecHits->begin(); it != HBHERecHits->end(); ++it) {
      totene += it->energy();
      hcalene += it->energy();
    }
  }

  if (totene > EnergyCutTot)
    acceptedtot = true;
  if (ecalene > EnergyCutEcal)
    acceptedEcal = true;
  if (hcalene > EnergyCutHcal)
    acceptedHcal = true;

  accepted = acceptedtot | acceptedEcal | acceptedHcal;

  if (accepted) {
    edm::LogVerbatim("BeamSplash") << "!!!!!!!BeamSplash!!!!!!!: run:" << irun << " event:" << ievt << " ls:" << ils
                                   << " bx= " << ibx << " totene=" << totene << " ecalene=" << ecalene
                                   << " hcalene=" << hcalene;
    std::cout << "!!!!!!!BeamSplash!!!!!!!: run:" << irun << " event:" << ievt << " ls:" << ils << " bx= " << ibx
              << " totene=" << totene << " ecalene=" << ecalene << " hcalene=" << hcalene << std::endl;
  }

  if (applyfilter)
    return accepted;
  else
    return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSplash);
