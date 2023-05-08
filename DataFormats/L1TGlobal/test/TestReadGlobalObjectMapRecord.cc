// -*- C++ -*-
//
// Package:    DataFormats/L1TGlobal
// Class:      TestReadGlobalObjectMapRecord
//
/**\class edmtest::TestReadGlobalObjectMapRecord
  Description: Used as part of tests that ensure the GlobalObjectMapRecord
  data format can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time the GlobalObjectMapRecord persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of the data format.
*/
// Original Author:  W. David Dagenhart
//         Created:  3 May 2023

#include "DataFormats/L1TGlobal/interface/GlobalObject.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapFwd.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <string>
#include <vector>

namespace edmtest {

  class TestReadGlobalObjectMapRecord : public edm::global::EDAnalyzer<> {
  public:
    TestReadGlobalObjectMapRecord(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    void throwWithMessage(const char*) const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    // These expected values are meaningless other than we use them
    // to check that values read from persistent storage match the values
    // we know were written.
    std::vector<std::string> expectedAlgoNames_;
    std::vector<int> expectedAlgoBitNumbers_;
    std::vector<int> expectedAlgoGtlResults_;
    std::vector<std::string> expectedTokenNames0_;
    std::vector<int> expectedTokenNumbers0_;
    std::vector<int> expectedTokenResults0_;
    std::vector<std::string> expectedTokenNames3_;
    std::vector<int> expectedTokenNumbers3_;
    std::vector<int> expectedTokenResults3_;
    int expectedFirstElement_;
    int expectedElementDelta_;
    int expectedFinalValue_;

    edm::EDGetTokenT<GlobalObjectMapRecord> globalObjectMapRecordToken_;
  };

  TestReadGlobalObjectMapRecord::TestReadGlobalObjectMapRecord(edm::ParameterSet const& iPSet)
      : expectedAlgoNames_(iPSet.getParameter<std::vector<std::string>>("expectedAlgoNames")),
        expectedAlgoBitNumbers_(iPSet.getParameter<std::vector<int>>("expectedAlgoBitNumbers")),
        expectedAlgoGtlResults_(iPSet.getParameter<std::vector<int>>("expectedAlgoGtlResults")),
        expectedTokenNames0_(iPSet.getParameter<std::vector<std::string>>("expectedTokenNames0")),
        expectedTokenNumbers0_(iPSet.getParameter<std::vector<int>>("expectedTokenNumbers0")),
        expectedTokenResults0_(iPSet.getParameter<std::vector<int>>("expectedTokenResults0")),
        expectedTokenNames3_(iPSet.getParameter<std::vector<std::string>>("expectedTokenNames3")),
        expectedTokenNumbers3_(iPSet.getParameter<std::vector<int>>("expectedTokenNumbers3")),
        expectedTokenResults3_(iPSet.getParameter<std::vector<int>>("expectedTokenResults3")),
        expectedFirstElement_(iPSet.getParameter<int>("expectedFirstElement")),
        expectedElementDelta_(iPSet.getParameter<int>("expectedElementDelta")),
        expectedFinalValue_(iPSet.getParameter<int>("expectedFinalValue")),

        globalObjectMapRecordToken_(consumes(iPSet.getParameter<edm::InputTag>("globalObjectMapRecordTag"))) {
    if (expectedAlgoNames_.size() != expectedAlgoBitNumbers_.size() ||
        expectedAlgoNames_.size() != expectedAlgoGtlResults_.size() ||
        expectedAlgoNames_.size() != expectedTokenNames0_.size() ||
        expectedAlgoNames_.size() != expectedTokenNumbers0_.size() ||
        expectedAlgoNames_.size() != expectedTokenResults0_.size() ||
        expectedAlgoNames_.size() != expectedTokenNames3_.size() ||
        expectedAlgoNames_.size() != expectedTokenNumbers3_.size() ||
        expectedAlgoNames_.size() != expectedTokenResults3_.size()) {
      throw cms::Exception("TestFailure") << "TestWriteGlobalObjectMapRecord, test configuration error: "
                                             "expectedAlgoNames, expectedAlgoBitNumbers, expectedAlgoGtlResults, "
                                             "expectedTokenNames0_, expectedTokenNumbers0_, expectedTokenResults0_, "
                                             "expectedTokenNames3_, expectedTokenNumbers3_, and expectedTokenResults3_ "
                                             "should have the same size.";
    }
  }

  void TestReadGlobalObjectMapRecord::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& globalObjectMapRecord = iEvent.get(globalObjectMapRecordToken_);

    std::vector<GlobalObjectMap> const& globalObjectMaps = globalObjectMapRecord.gtObjectMap();

    if (expectedAlgoNames_.size() != globalObjectMaps.size()) {
      throwWithMessage("globalObjectMaps does not have expected size");
    }

    unsigned int index = 0;
    for (auto const& globalObjectMap : globalObjectMaps) {
      if (globalObjectMap.algoName() != expectedAlgoNames_[index]) {
        throwWithMessage("algoName does not have expected value");
      }
      if (globalObjectMap.algoBitNumber() != expectedAlgoBitNumbers_[index]) {
        throwWithMessage("algoBitNumber does not have expected value");
      }
      if (globalObjectMap.algoGtlResult() != static_cast<bool>(expectedAlgoGtlResults_[index])) {
        throwWithMessage("algoGtlResult does not have expected value");
      }

      std::vector<GlobalLogicParser::OperandToken> const& operandTokens = globalObjectMap.operandTokenVector();
      if (operandTokens[0].tokenName != expectedTokenNames0_[index]) {
        throwWithMessage("tokenName0 does not have expected value");
      }
      if (operandTokens[0].tokenNumber != expectedTokenNumbers0_[index]) {
        throwWithMessage("tokenNumber0 does not have expected value");
      }
      if (operandTokens[0].tokenResult != static_cast<bool>(expectedTokenResults0_[index])) {
        throwWithMessage("tokenResult0 does not have expected value");
      }
      if (operandTokens[3].tokenName != expectedTokenNames3_[index]) {
        throwWithMessage("tokenName3 does not have expected value");
      }
      if (operandTokens[3].tokenNumber != expectedTokenNumbers3_[index]) {
        throwWithMessage("tokenNumber3 does not have expected value");
      }
      if (operandTokens[3].tokenResult != static_cast<bool>(expectedTokenResults3_[index])) {
        throwWithMessage("tokenResult3 does not have expected value");
      }

      int expectedValue = expectedFirstElement_;
      for (auto const& combinationsInCond : globalObjectMap.combinationVector()) {
        for (auto const& singleCombInCond : combinationsInCond) {
          for (auto const& value : singleCombInCond) {
            if (value != expectedValue) {
              throwWithMessage("element in inner combination vector does have expected value");
            }
            expectedValue += expectedElementDelta_;
          }
        }
      }

      for (auto const& l1tObjectTypeInCond : globalObjectMap.objectTypeVector()) {
        for (auto const& globalObject : l1tObjectTypeInCond) {
          if (static_cast<int>(globalObject) != (expectedValue % 28)) {
            throwWithMessage("globalObject does have expected value");
          }
          expectedValue += expectedElementDelta_;
        }
      }
      if (expectedValue != expectedFinalValue_) {
        throw cms::Exception("TestFailure")
            << "final value = " << expectedValue << " which does not match the expected value of "
            << expectedFinalValue_ << " this might mean the vectors did not contain the expected number of elements";
      }
      ++index;
    }
  }

  void TestReadGlobalObjectMapRecord::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadGlobalObjectMapRecord::analyze, " << msg;
  }

  void TestReadGlobalObjectMapRecord::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<std::string>>("expectedAlgoNames");
    desc.add<std::vector<int>>("expectedAlgoBitNumbers");
    desc.add<std::vector<int>>("expectedAlgoGtlResults");
    desc.add<std::vector<std::string>>("expectedTokenNames0");
    desc.add<std::vector<int>>("expectedTokenNumbers0");
    desc.add<std::vector<int>>("expectedTokenResults0");
    desc.add<std::vector<std::string>>("expectedTokenNames3");
    desc.add<std::vector<int>>("expectedTokenNumbers3");
    desc.add<std::vector<int>>("expectedTokenResults3");
    desc.add<int>("expectedFirstElement");
    desc.add<int>("expectedElementDelta");
    desc.add<int>("expectedFinalValue");
    desc.add<edm::InputTag>("globalObjectMapRecordTag");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadGlobalObjectMapRecord;
DEFINE_FWK_MODULE(TestReadGlobalObjectMapRecord);
