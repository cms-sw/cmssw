// -*- C++ -*-
//
// Package:    IOPool/Input
// Class:      SchemaEvolutionTestWrite
//
/**\class edmtest::SchemaEvolutionTestWrite
  Description: Used as part of tests of ROOT's schema evolution
  features. These features allow reading a persistent object
  whose data format has changed since it was written.
*/
// Original Author:  W. David Dagenhart
//         Created:  24 July 2023

#include "DataFormats/TestObjects/interface/VectorVectorTop.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>
#include <utility>
#include <vector>

namespace edmtest {

  class SchemaEvolutionTestWrite : public edm::global::EDProducer<> {
  public:
    SchemaEvolutionTestWrite(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produceVectorVector(edm::Event&) const;
    void produceVectorVectorNonSplit(edm::Event&) const;

    void throwWithMessage(const char*) const;

    const std::vector<int> testIntegralValues_;
    const edm::EDPutTokenT<VectorVectorTop> vectorVectorPutToken_;
    const edm::EDPutTokenT<VectorVectorTopNonSplit> vectorVectorNonSplitPutToken_;
  };

  SchemaEvolutionTestWrite::SchemaEvolutionTestWrite(edm::ParameterSet const& iPSet)
      : testIntegralValues_(iPSet.getParameter<std::vector<int>>("testIntegralValues")),
        vectorVectorPutToken_(produces()),
        vectorVectorNonSplitPutToken_(produces()) {
    if (testIntegralValues_.size() != 15) {
      throwWithMessage("testIntegralValues must have 15 elements and it does not");
    }
  }

  void SchemaEvolutionTestWrite::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    // Fill test objects. Make sure all the containers inside
    // of them have something in them (not empty). The values are meaningless.
    // We will later check that after writing these objects to persistent storage
    // and then reading them in a later process we obtain matching values for
    // all this content.

    produceVectorVector(iEvent);
    produceVectorVectorNonSplit(iEvent);
  }

  void SchemaEvolutionTestWrite::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<int>>("testIntegralValues");
    descriptions.addDefault(desc);
  }

  void SchemaEvolutionTestWrite::produceVectorVector(edm::Event& iEvent) const {
    auto vectorVector = std::make_unique<VectorVectorTop>();
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    vectorVector->outerVector_.reserve(vectorSize);
    for (unsigned int i = 0; i < vectorSize; ++i) {
      int iOffset = static_cast<int>(iEvent.id().event() + i);
      VectorVectorMiddle middleVector;
      middleVector.middleVector_.reserve(vectorSize);
      for (int j = 0; j < static_cast<int>(vectorSize); ++j) {
        SchemaEvolutionChangeOrder changeOrder(testIntegralValues_[1] + iOffset + j * 11,
                                               testIntegralValues_[1] + iOffset + j * 101);
        SchemaEvolutionAddMember addMember(testIntegralValues_[2] + iOffset + j * 12,
                                           testIntegralValues_[2] + iOffset + j * 102,
                                           testIntegralValues_[2] + iOffset + j * 1002);
        SchemaEvolutionRemoveMember removeMember(testIntegralValues_[3] + iOffset + j * 13,
                                                 testIntegralValues_[3] + iOffset + j * 103);
        SchemaEvolutionMoveToBase moveToBase(testIntegralValues_[4] + iOffset + j * 14,
                                             testIntegralValues_[4] + iOffset + j * 104,
                                             testIntegralValues_[4] + iOffset + j * 1004,
                                             testIntegralValues_[4] + iOffset + j * 10004);
        SchemaEvolutionChangeType changeType(testIntegralValues_[5] + iOffset + j * 15,
                                             testIntegralValues_[5] + iOffset + j * 105);
        SchemaEvolutionAddBase addBase(testIntegralValues_[6] + iOffset + j * 16,
                                       testIntegralValues_[6] + iOffset + j * 106,
                                       testIntegralValues_[6] + iOffset + j * 1006);
        SchemaEvolutionPointerToMember pointerToMember(testIntegralValues_[7] + iOffset + j * 17,
                                                       testIntegralValues_[7] + iOffset + j * 107,
                                                       testIntegralValues_[7] + iOffset + j * 1007);
        SchemaEvolutionPointerToUniquePtr pointerToUniquePtr(testIntegralValues_[8] + iOffset + j * 18,
                                                             testIntegralValues_[8] + iOffset + j * 108,
                                                             testIntegralValues_[8] + iOffset + j * 1008);
        SchemaEvolutionCArrayToStdArray cArrayToStdArray(testIntegralValues_[9] + iOffset + j * 19,
                                                         testIntegralValues_[9] + iOffset + j * 109,
                                                         testIntegralValues_[9] + iOffset + j * 1009);
        // This is commented out because schema evolution fails for this case
        // SchemaEvolutionCArrayToStdVector cArrayToStdVector(testIntegralValues_[10] + iOffset + j * 20, testIntegralValues_[10] + iOffset + j * 110, testIntegralValues_[10] + iOffset + j * 1010);
        SchemaEvolutionVectorToList vectorToList(testIntegralValues_[11] + iOffset + j * 21,
                                                 testIntegralValues_[11] + iOffset + j * 111,
                                                 testIntegralValues_[11] + iOffset + j * 1011);
        SchemaEvolutionMapToUnorderedMap mapToUnorderedMap(testIntegralValues_[12] + iOffset + j * 22,
                                                           testIntegralValues_[12] + iOffset + j * 112,
                                                           testIntegralValues_[12] + iOffset + j * 1012 + 1012,
                                                           testIntegralValues_[12] + iOffset + j * 10012,
                                                           testIntegralValues_[12] + iOffset + j * 100012 + 100012,
                                                           testIntegralValues_[12] + iOffset + j * 1000012);

        middleVector.middleVector_.emplace_back(testIntegralValues_[0] + iOffset + j * 10,
                                                testIntegralValues_[0] + iOffset + j * 100,
                                                changeOrder,
                                                addMember,
                                                removeMember,
                                                moveToBase,
                                                changeType,
                                                addBase,
                                                pointerToMember,
                                                pointerToUniquePtr,
                                                cArrayToStdArray,
                                                // cArrayToStdVector,
                                                vectorToList,
                                                mapToUnorderedMap);
      }
      vectorVector->outerVector_.push_back(std::move(middleVector));
    }
    iEvent.put(vectorVectorPutToken_, std::move(vectorVector));
  }

  void SchemaEvolutionTestWrite::produceVectorVectorNonSplit(edm::Event& iEvent) const {
    auto vectorVector = std::make_unique<VectorVectorTopNonSplit>();
    VectorVectorMiddleNonSplit middleVector;
    middleVector.middleVector_.emplace_back(testIntegralValues_[13], testIntegralValues_[14]);
    vectorVector->outerVector_.push_back(middleVector);
    iEvent.put(vectorVectorNonSplitPutToken_, std::move(vectorVector));
  }

  void SchemaEvolutionTestWrite::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "SchemaEvolutionTestWrite constructor, test configuration error, " << msg;
  }

}  // namespace edmtest

using edmtest::SchemaEvolutionTestWrite;
DEFINE_FWK_MODULE(SchemaEvolutionTestWrite);
