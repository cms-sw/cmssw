// -*- C++ -*-
//
// Package:   ECALActivity
// Class:     ECALActivity
//
//
// Original Author:  Luca Malgeri

#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "DPGAnalysis/Skims/interface/ECALActivity.h"

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

using namespace edm;
using namespace std;

ECALActivity::ECALActivity(const edm::ParameterSet& iConfig) {
  EBRecHitCollection_ = iConfig.getParameter<edm::InputTag>("ebrechitcollection");
  EERecHitCollection_ = iConfig.getParameter<edm::InputTag>("ebrechitcollection");

  EBnum = iConfig.getUntrackedParameter<int>("EBnum");
  EBthresh = iConfig.getUntrackedParameter<double>("EBthresh");
  EEnum = iConfig.getUntrackedParameter<int>("EEnum");
  EEthresh = iConfig.getUntrackedParameter<double>("EEthresh");
  ETOTnum = iConfig.getUntrackedParameter<int>("ETOTnum");
  ETOTthresh = iConfig.getUntrackedParameter<double>("ETOTthresh");
  applyfilter = iConfig.getUntrackedParameter<bool>("applyfilter", true);
}

ECALActivity::~ECALActivity() {}

bool ECALActivity::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool accepted = false;
  bool eb = false;
  bool ee = false;
  bool etot = false;

  //int ievt = iEvent.id().event();
  //int irun = iEvent.id().run();
  //int ils = iEvent.luminosityBlock();

  int ebabovethresh = 0;
  int eeabovethresh = 0;
  int etotabovethresh = 0;

  Handle<EBRecHitCollection> pEBRecHits;
  Handle<EERecHitCollection> pEERecHits;

  const EBRecHitCollection* EBRecHits = nullptr;
  const EERecHitCollection* EERecHits = nullptr;

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

  // now loop over them
  if (EBRecHits) {
    for (EBRecHitCollection::const_iterator it = EBRecHits->begin(); it != EBRecHits->end(); ++it) {
      if (it->energy() > EBthresh)
        ebabovethresh++;
      if (it->energy() > ETOTthresh)
        etotabovethresh++;
    }
  }
  if (EERecHits) {
    for (EERecHitCollection::const_iterator it = EERecHits->begin(); it != EERecHits->end(); ++it) {
      if (it->energy() > EEthresh)
        eeabovethresh++;
      if (it->energy() > ETOTthresh)
        etotabovethresh++;
    }
  }

  if (ebabovethresh >= EBnum)
    eb = true;
  if (eeabovethresh >= EEnum)
    ee = true;
  if (etotabovethresh >= ETOTnum)
    etot = true;

  accepted = eb | ee | etot;

  if (applyfilter)
    return accepted;
  else
    return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ECALActivity);
