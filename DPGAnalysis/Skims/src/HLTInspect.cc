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
#include "DPGAnalysis/Skims/interface/HLTInspect.h"

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

HLTInspect::HLTInspect(const edm::ParameterSet& iConfig) {
  hlTriggerResults_ = iConfig.getParameter<edm::InputTag>("HLTriggerResults");
  init_ = false;
}

HLTInspect::~HLTInspect() {}
void HLTInspect::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {
  int ievt = iEvent.id().event();
  int irun = iEvent.id().run();
  int ils = iEvent.luminosityBlock();
  int bx = iEvent.bunchCrossing();
  //
  // trigger type
  //
  int trigger_type = -1;
  if (iEvent.isRealData())
    trigger_type = iEvent.experimentType();

  //hlt info
  edm::Handle<TriggerResults> HLTR;
  iEvent.getByLabel(hlTriggerResults_, HLTR);

  if (HLTR.isValid() == false) {
    std::cout << " HLTInspect Error - Could not access Results with name " << hlTriggerResults_ << std::endl;
  }
  if (HLTR.isValid()) {
    if (!init_) {
      init_ = true;
      const edm::TriggerNames& triggerNames = iEvent.triggerNames(*HLTR);
      hlNames_ = triggerNames.triggerNames();
    }
    std::cout << "HLTInspect: Run " << irun << " Ev " << ievt << " LB " << ils << " BX " << bx << " Type "
              << trigger_type << " Acc: ";
    const unsigned int n(hlNames_.size());
    for (unsigned int i = 0; i != n; ++i) {
      if (HLTR->accept(i)) {
        std::cout << hlNames_[i] << ",";
      }
    }
    std::cout << std::endl;
  }
}
//define this as a plug-in
DEFINE_FWK_MODULE(HLTInspect);
