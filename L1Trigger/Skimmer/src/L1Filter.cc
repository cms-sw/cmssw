#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EDFilter.h"
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

class L1Filter : public edm::EDFilter {
public:
  explicit L1Filter(edm::ParameterSet const &);

  ~L1Filter() override;

  bool filter(edm::Event &e, edm::EventSetup const &c) override;
  void endJob() override;

private:
  edm::InputTag inputTag_;
  bool useAODRecord_;
  bool useFinalDecision_;
  std::vector<std::string> algos_;
};

L1Filter::L1Filter(const edm::ParameterSet &ps)
    : inputTag_(ps.getParameter<edm::InputTag>("inputTag")),
      useAODRecord_(ps.getParameter<bool>("useAODRecord")),
      useFinalDecision_(ps.getParameter<bool>("useFinalDecision")),
      algos_(ps.getParameter<std::vector<std::string>>("algorithms")) {}

L1Filter::~L1Filter() {}

bool L1Filter::filter(edm::Event &iEvent, edm::EventSetup const &evSetup) {
  // get menu
  edm::ESHandle<L1GtTriggerMenu> menuRcd;
  evSetup.get<L1GtTriggerMenuRcd>().get(menuRcd);
  const L1GtTriggerMenu *menu = menuRcd.product();

  bool passed = false;
  std::vector<std::string>::const_iterator algo;

  if (useAODRecord_) {
    edm::Handle<L1GlobalTriggerRecord> gtRecord;
    iEvent.getByLabel(inputTag_, gtRecord);
    const DecisionWord dWord = gtRecord->decisionWord();

    if (useFinalDecision_)
      passed = gtRecord->decision();
    else {
      for (algo = algos_.begin(); algo != algos_.end(); ++algo) {
        passed |= menu->gtAlgorithmResult((*algo), dWord);
      }
    }
  } else {
    edm::Handle<L1GlobalTriggerReadoutRecord> gtRecord;
    iEvent.getByLabel(inputTag_, gtRecord);
    const DecisionWord dWord = gtRecord->decisionWord();

    if (useFinalDecision_)
      passed = gtRecord->decision();
    else {
      for (algo = algos_.begin(); algo != algos_.end(); ++algo) {
        passed |= menu->gtAlgorithmResult((*algo), dWord);
      }
    }
  }

  return passed;
}

void L1Filter::endJob() {}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(L1Filter);
