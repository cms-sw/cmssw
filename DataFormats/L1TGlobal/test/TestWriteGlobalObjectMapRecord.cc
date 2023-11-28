// -*- C++ -*-
//
// Package:    DataFormats/L1TGlobal
// Class:      TestWriteGlobalObjectMapRecord
//
/**\class edmtest::TestWriteGlobalObjectMapRecord
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

#include "DataFormats/L1TGlobal/interface/GlobalObjectMap.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace edmtest {

  class TestWriteGlobalObjectMapRecord : public edm::global::EDProducer<> {
  public:
    TestWriteGlobalObjectMapRecord(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    unsigned int nGlobalObjectMaps_;
    std::vector<std::string> algoNames_;
    std::vector<int> algoBitNumbers_;
    std::vector<int> algoResults_;
    std::vector<std::string> tokenNames0_;
    std::vector<int> tokenNumbers0_;
    std::vector<int> tokenResults0_;
    std::vector<std::string> tokenNames3_;
    std::vector<int> tokenNumbers3_;
    std::vector<int> tokenResults3_;
    unsigned int nElements1_;
    unsigned int nElements2_;
    unsigned int nElements3_;
    int firstElement_;
    int elementDelta_;

    edm::EDPutTokenT<GlobalObjectMapRecord> globalObjectMapRecordPutToken_;
  };

  TestWriteGlobalObjectMapRecord::TestWriteGlobalObjectMapRecord(edm::ParameterSet const& iPSet)
      : nGlobalObjectMaps_(iPSet.getParameter<unsigned int>("nGlobalObjectMaps")),
        algoNames_(iPSet.getParameter<std::vector<std::string>>("algoNames")),
        algoBitNumbers_(iPSet.getParameter<std::vector<int>>("algoBitNumbers")),
        algoResults_(iPSet.getParameter<std::vector<int>>("algoResults")),
        tokenNames0_(iPSet.getParameter<std::vector<std::string>>("tokenNames0")),
        tokenNumbers0_(iPSet.getParameter<std::vector<int>>("tokenNumbers0")),
        tokenResults0_(iPSet.getParameter<std::vector<int>>("tokenResults0")),
        tokenNames3_(iPSet.getParameter<std::vector<std::string>>("tokenNames3")),
        tokenNumbers3_(iPSet.getParameter<std::vector<int>>("tokenNumbers3")),
        tokenResults3_(iPSet.getParameter<std::vector<int>>("tokenResults3")),
        nElements1_(iPSet.getParameter<unsigned int>("nElements1")),
        nElements2_(iPSet.getParameter<unsigned int>("nElements2")),
        nElements3_(iPSet.getParameter<unsigned int>("nElements3")),
        firstElement_(iPSet.getParameter<int>("firstElement")),
        elementDelta_(iPSet.getParameter<int>("elementDelta")),
        globalObjectMapRecordPutToken_(produces()) {
    if (algoNames_.size() != nGlobalObjectMaps_ || algoBitNumbers_.size() != nGlobalObjectMaps_ ||
        algoResults_.size() != nGlobalObjectMaps_ || tokenNames0_.size() != nGlobalObjectMaps_ ||
        tokenNumbers0_.size() != nGlobalObjectMaps_ || tokenResults0_.size() != nGlobalObjectMaps_ ||
        tokenNames3_.size() != nGlobalObjectMaps_ || tokenNumbers3_.size() != nGlobalObjectMaps_ ||
        tokenResults3_.size() != nGlobalObjectMaps_) {
      throw cms::Exception("TestFailure")
          << "TestWriteGlobalObjectMapRecord, test configuration error: "
             "algoNames, algoBitNumbers, algoResults, tokenNames0, tokenNumbers0, tokenResults0, "
             "tokenNames3, tokenNumbers3, and tokenResults3 should have size equal to nGlobalObjectMaps";
    }
  }

  void TestWriteGlobalObjectMapRecord::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    // Fill a GlobalObjectMapRecord. Make sure all the containers inside
    // of it have something in them (not empty). The values are meaningless.
    // We will later check that after writing this object to persistent storage
    // and then reading it in a later process we obtain matching values for
    // all this content.

    std::vector<GlobalObjectMap> globalObjectMapVector(nGlobalObjectMaps_);
    for (unsigned int i = 0; i < nGlobalObjectMaps_; ++i) {
      GlobalObjectMap& globalObjectMap = globalObjectMapVector[i];
      globalObjectMap.setAlgoName(algoNames_[i]);
      globalObjectMap.setAlgoBitNumber(algoBitNumbers_[i]);
      globalObjectMap.setAlgoGtlResult(static_cast<bool>(algoResults_[i]));

      std::vector<GlobalLogicParser::OperandToken> tokenVector;
      // We will later check elements 0 and 3 after writing to persistent
      // storage and then reading (seemed like checking two elements
      // would be enough to verify the reading and writing was working properly,
      // the selection of 0 and 3 was meaningless and rather arbitrary)
      GlobalLogicParser::OperandToken token0{tokenNames0_[i], tokenNumbers0_[i], static_cast<bool>(tokenResults0_[i])};
      GlobalLogicParser::OperandToken token3{tokenNames3_[i], tokenNumbers3_[i], static_cast<bool>(tokenResults3_[i])};
      tokenVector.push_back(token0);  // We check this element
      tokenVector.push_back(token0);
      tokenVector.push_back(token0);
      tokenVector.push_back(token3);  // we also check this element
      globalObjectMap.swapOperandTokenVector(tokenVector);

      // We fill these with an arithmetic sequence of values.
      // Again, this is just an arbitrary meaningless test pattern.
      // The only purpose is to later check that when
      // we read we get values that match what we wrote.
      int value = firstElement_;
      std::vector<CombinationsInCond> combinationsInCondVector;
      for (unsigned int i = 0; i < nElements1_; ++i) {
        CombinationsInCond combinationsInCond;
        for (unsigned int j = 0; j < nElements2_; ++j) {
          SingleCombInCond singleCombInCond;
          for (unsigned int k = 0; k < nElements3_; ++k) {
            singleCombInCond.push_back(value);
            value += elementDelta_;
          }
          combinationsInCond.push_back(std::move(singleCombInCond));
        }
        combinationsInCondVector.push_back(combinationsInCond);
      }
      globalObjectMap.swapCombinationVector(combinationsInCondVector);

      std::vector<L1TObjectTypeInCond> objectTypeVector;
      for (unsigned int i = 0; i < nElements1_; ++i) {
        L1TObjectTypeInCond globalObjects;
        for (unsigned int j = 0; j < nElements2_; ++j) {
          globalObjects.push_back(static_cast<l1t::GlobalObject>(value % 28));
          value += elementDelta_;
        }
        objectTypeVector.push_back(std::move(globalObjects));
      }
      globalObjectMap.swapObjectTypeVector(objectTypeVector);
    }

    auto globalObjectMapRecord = std::make_unique<GlobalObjectMapRecord>();
    globalObjectMapRecord->swapGtObjectMap(globalObjectMapVector);
    iEvent.put(globalObjectMapRecordPutToken_, std::move(globalObjectMapRecord));
  }

  void TestWriteGlobalObjectMapRecord::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<unsigned int>("nGlobalObjectMaps");
    desc.add<std::vector<std::string>>("algoNames");
    desc.add<std::vector<int>>("algoBitNumbers");
    desc.add<std::vector<int>>("algoResults");

    desc.add<std::vector<std::string>>("tokenNames0");
    desc.add<std::vector<int>>("tokenNumbers0");
    desc.add<std::vector<int>>("tokenResults0");
    desc.add<std::vector<std::string>>("tokenNames3");
    desc.add<std::vector<int>>("tokenNumbers3");
    desc.add<std::vector<int>>("tokenResults3");

    desc.add<unsigned int>("nElements1");
    desc.add<unsigned int>("nElements2");
    desc.add<unsigned int>("nElements3");
    desc.add<int>("firstElement");
    desc.add<int>("elementDelta");

    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestWriteGlobalObjectMapRecord;
DEFINE_FWK_MODULE(TestWriteGlobalObjectMapRecord);
