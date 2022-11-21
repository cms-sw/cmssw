#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"

#include <iostream>
#include <string>

class L1Filter : public edm::global::EDFilter<> {
public:
  explicit L1Filter(edm::ParameterSet const &);

  bool filter(edm::StreamID, edm::Event &e, edm::EventSetup const &c) const override;

private:
  edm::EDGetTokenT<L1GlobalTriggerRecord> triggerToken_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> triggerReadoutToken_;
  const edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> menuToken_;
  const bool useAODRecord_;
  const bool useFinalDecision_;
  const std::vector<std::string> algos_;
};

L1Filter::L1Filter(const edm::ParameterSet &ps)
    : menuToken_(esConsumes()),
      useAODRecord_(ps.getParameter<bool>("useAODRecord")),
      useFinalDecision_(ps.getParameter<bool>("useFinalDecision")),
      algos_(ps.getParameter<std::vector<std::string>>("algorithms")) {
  auto inputTag = ps.getParameter<edm::InputTag>("inputTag");

  if (useAODRecord_) {
    triggerToken_ = consumes(inputTag);
  } else {
    triggerReadoutToken_ = consumes(inputTag);
  }
}

bool L1Filter::filter(edm::StreamID, edm::Event &iEvent, edm::EventSetup const &evSetup) const {
  // get menu
  const L1GtTriggerMenu &menu = evSetup.getData(menuToken_);

  bool passed = false;
  std::vector<std::string>::const_iterator algo;

  if (useAODRecord_) {
    L1GlobalTriggerRecord const &gtRecord = iEvent.get(triggerToken_);

    if (useFinalDecision_)
      passed = gtRecord.decision();
    else {
      const DecisionWord dWord = gtRecord.decisionWord();
      for (algo = algos_.begin(); algo != algos_.end(); ++algo) {
        passed |= menu.gtAlgorithmResult((*algo), dWord);
      }
    }
  } else {
    L1GlobalTriggerReadoutRecord const &gtRecord = iEvent.get(triggerReadoutToken_);

    if (useFinalDecision_)
      passed = gtRecord.decision();
    else {
      const DecisionWord &dWord = gtRecord.decisionWord();
      for (algo = algos_.begin(); algo != algos_.end(); ++algo) {
        passed |= menu.gtAlgorithmResult((*algo), dWord);
      }
    }
  }

  return passed;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(L1Filter);
