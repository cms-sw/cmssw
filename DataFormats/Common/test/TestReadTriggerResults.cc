// -*- C++ -*-
//
// Package:    DataFormats/Common
// Class:      TestReadTriggerResults
//
/**\class edmtest::TestReadTriggerResults
  Description: Used as part of tests that ensure the TriggerResults
  data format can be persistently written and in a subsequent process
  read. First, this is done using the current release version. In
  addition, the output file of the write process should be saved
  permanently each time its format changes. In unit tests, we read
  each of those saved files to verify that all future releases can
  read RAW data formats and Scouting data formats.
*/
// Original Author:  W. David Dagenhart
//         Created:  18 April 2023

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>

namespace edmtest {

  class TestReadTriggerResults : public edm::global::EDAnalyzer<> {
  public:
    TestReadTriggerResults(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    void throwWithMessage(const char*) const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    std::string expectedParameterSetID_;
    std::vector<std::string> expectedNames_;
    std::vector<unsigned int> expectedHLTStates_;
    std::vector<unsigned int> expectedModuleIndexes_;
    edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  };

  TestReadTriggerResults::TestReadTriggerResults(edm::ParameterSet const& iPSet)
      : expectedParameterSetID_(iPSet.getParameter<std::string>("expectedParameterSetID")),
        expectedNames_(iPSet.getParameter<std::vector<std::string>>("expectedNames")),
        expectedHLTStates_(iPSet.getParameter<std::vector<unsigned int>>("expectedHLTStates")),
        expectedModuleIndexes_(iPSet.getParameter<std::vector<unsigned int>>("expectedModuleIndexes")),
        triggerResultsToken_(consumes(iPSet.getParameter<edm::InputTag>("triggerResultsTag"))) {}

  void TestReadTriggerResults::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& triggerResults = iEvent.get(triggerResultsToken_);
    std::string parameterSetID;
    triggerResults.parameterSetID().toString(parameterSetID);
    if (parameterSetID != expectedParameterSetID_) {
      throwWithMessage("parameterSetID does not match expected value");
    }
    if (triggerResults.getTriggerNames() != expectedNames_) {
      throwWithMessage("names vector does not include expected values");
    }
    if (expectedHLTStates_.size() != expectedModuleIndexes_.size()) {
      throwWithMessage(
          "test configuration error, expectedHLTStates and expectedModuleIndexes should have the same size");
    }
    if (triggerResults.size() != expectedHLTStates_.size()) {
      throwWithMessage("paths has unexpected size");
    }
    for (unsigned int i = 0; i < expectedHLTStates_.size(); ++i) {
      if (static_cast<unsigned int>(triggerResults.state(i)) != expectedHLTStates_[i]) {
        throwWithMessage("state has unexpected value");
      }
      if (triggerResults.index(i) != expectedModuleIndexes_[i]) {
        throwWithMessage("module index has unexpected value");
      }
    }
  }

  void TestReadTriggerResults::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadTriggerResults::analyze, " << msg;
  }

  void TestReadTriggerResults::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("expectedParameterSetID");
    desc.add<std::vector<std::string>>("expectedNames");
    desc.add<std::vector<unsigned int>>("expectedHLTStates");
    desc.add<std::vector<unsigned int>>("expectedModuleIndexes");
    desc.add<edm::InputTag>("triggerResultsTag");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadTriggerResults;
DEFINE_FWK_MODULE(TestReadTriggerResults);
