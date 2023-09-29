// -*- C++ -*-
//
// Package:    DataFormats/Common
// Class:      TestWriteTriggerResults
//
/**\class edmtest::TestWriteTriggerResults
  Description: Used as part of tests that ensure the TriggerResults
  data format can be persistently written and in a subsequent process
  read. First, this is done using the current release version. In
  addition, the output file of the write process should be saved
  permanently each time its format changes. In unit tests, we read
  each of those saved files to verify that all future releases can
  read all versions of RAW data formats and Scouting data formats.
*/
// Original Author:  W. David Dagenhart
//         Created:  20 April 2023

#include "DataFormats/Common/interface/HLTGlobalStatus.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include <cassert>
#include <memory>
#include <string>
#include <vector>

namespace edmtest {

  class TestWriteTriggerResults : public edm::global::EDProducer<> {
  public:
    TestWriteTriggerResults(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    std::string parameterSetID_;
    std::vector<std::string> names_;
    std::vector<unsigned int> hltStates_;
    std::vector<unsigned int> moduleIndexes_;
    edm::EDPutTokenT<edm::TriggerResults> triggerResultsPutToken_;
  };

  TestWriteTriggerResults::TestWriteTriggerResults(edm::ParameterSet const& iPSet)
      : parameterSetID_(iPSet.getParameter<std::string>("parameterSetID")),
        names_(iPSet.getParameter<std::vector<std::string>>("names")),
        hltStates_(iPSet.getParameter<std::vector<unsigned int>>("hltStates")),
        moduleIndexes_(iPSet.getParameter<std::vector<unsigned int>>("moduleIndexes")),
        triggerResultsPutToken_(produces()) {}

  void TestWriteTriggerResults::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    edm::HLTGlobalStatus hltGlobalStatus(hltStates_.size());
    for (unsigned int i = 0; i < hltStates_.size(); ++i) {
      assert(i < moduleIndexes_.size());
      hltGlobalStatus[i] = edm::HLTPathStatus(static_cast<edm::hlt::HLTState>(hltStates_[i]), moduleIndexes_[i]);
    }
    edm::ParameterSetID parameterSetID(parameterSetID_);
    std::unique_ptr<edm::TriggerResults> result;
    if (names_.empty()) {
      // names_ will always be empty except in extremely old data or monte carlo files
      result = std::make_unique<edm::TriggerResults>(hltGlobalStatus, parameterSetID);
    } else {
      // If names is not empty, the ParameterSetID is not set and default constructed
      result = std::make_unique<edm::TriggerResults>(hltGlobalStatus, names_);
    }
    iEvent.put(triggerResultsPutToken_, std::move(result));
  }

  void TestWriteTriggerResults::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("parameterSetID");
    desc.add<std::vector<std::string>>("names");
    desc.add<std::vector<unsigned int>>("hltStates");
    desc.add<std::vector<unsigned int>>("moduleIndexes");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestWriteTriggerResults;
DEFINE_FWK_MODULE(TestWriteTriggerResults);
