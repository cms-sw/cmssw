// -*- C++ -*-
//
// Package:   BeamSplash
// Class:     BeamSPlash
//
//
// Original Author:  Luca Malgeri

#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "DPGAnalysis/Skims/interface/PhysDecl.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

using namespace edm;
using namespace std;

PhysDecl::PhysDecl(const edm::ParameterSet& iConfig) {
  applyfilter = iConfig.getUntrackedParameter<bool>("applyfilter", true);
  debugOn = iConfig.getUntrackedParameter<bool>("debugOn", false);
  hlTriggerResults_ = consumes<TriggerResults>(iConfig.getParameter<edm::InputTag>("HLTriggerResults"));
  gtDigis_ = consumes<L1GlobalTriggerReadoutRecord>(
      iConfig.getUntrackedParameter<edm::InputTag>("gtDigis", edm::InputTag("gtDigis")));
  init_ = false;
}

PhysDecl::~PhysDecl() {}

bool PhysDecl::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool accepted = false;

  int ievt = iEvent.id().event();
  int irun = iEvent.id().run();
  int ils = iEvent.luminosityBlock();
  int bx = iEvent.bunchCrossing();

  //hlt info
  edm::Handle<TriggerResults> HLTR;
  iEvent.getByToken(hlTriggerResults_, HLTR);

  if (HLTR.isValid()) {
    if (!init_) {
      init_ = true;
      const edm::TriggerNames& triggerNames = iEvent.triggerNames(*HLTR);
      hlNames_ = triggerNames.triggerNames();
    }
    if (debugOn) {
      std::cout << "HLT_debug: Run " << irun << " Ev " << ievt << " LB " << ils << " BX " << bx << " Acc: ";
      const unsigned int n(hlNames_.size());
      for (unsigned int i = 0; i != n; ++i) {
        if (HLTR->accept(i)) {
          std::cout << hlNames_[i] << ",";
        }
      }
      std::cout << std::endl;
    }
  }

  // trigger info

  edm::Handle<L1GlobalTriggerReadoutRecord> gtrr_handle;
  iEvent.getByToken(gtDigis_, gtrr_handle);
  L1GlobalTriggerReadoutRecord const* gtrr = gtrr_handle.product();

  L1GtFdlWord fdlWord = gtrr->gtFdlWord();
  //   std::cout << "phys decl. bit=" << fdlWord.physicsDeclared() << std::endl;
  if (fdlWord.physicsDeclared() == 1)
    accepted = true;

  if (debugOn) {
    std::cout << "PhysDecl_debug: Run " << irun << " Event " << ievt << " Lumi Block " << ils << " Bunch Crossing "
              << bx << " Accepted " << accepted << std::endl;
  }

  if (applyfilter)
    return accepted;
  else
    return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(PhysDecl);
