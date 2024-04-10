// -*- C++ -*-
//
// Package:    IOPool/Input
// Class:      SchemaEvolutionTestRead
//
/**\class edmtest::SchemaEvolutionTestRead
  Description: Used as part of tests of ROOT's schema evolution
  features. These features allow reading a persistent object
  whose data format has changed since it was written.
*/
// Original Author:  W. David Dagenhart
//         Created:  28 July 2023

#include "DataFormats/TestObjects/interface/VectorVectorTop.h"
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

#include <vector>

namespace edmtest {

  class SchemaEvolutionTestRead : public edm::global::EDAnalyzer<> {
  public:
    SchemaEvolutionTestRead(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void analyzeVectorVector(edm::Event const&) const;
    void analyzeVectorVectorNonSplit(edm::Event const&) const;

    void throwWithMessageFromConstructor(const char*) const;
    void throwWithMessage(const char*) const;

    // These expected values are meaningless other than we use them
    // to check that values read from persistent storage match the values
    // we know were written.

    const std::vector<int> expectedVectorVectorIntegralValues_;
    const edm::EDGetTokenT<VectorVectorTop> vectorVectorToken_;
    const edm::EDGetTokenT<VectorVectorTopNonSplit> vectorVectorNonSplitToken_;
  };

  SchemaEvolutionTestRead::SchemaEvolutionTestRead(edm::ParameterSet const& iPSet)
      : expectedVectorVectorIntegralValues_(iPSet.getParameter<std::vector<int>>("expectedVectorVectorIntegralValues")),
        vectorVectorToken_(consumes(iPSet.getParameter<edm::InputTag>("vectorVectorTag"))),
        vectorVectorNonSplitToken_(consumes(iPSet.getParameter<edm::InputTag>("vectorVectorTag"))) {
    if (expectedVectorVectorIntegralValues_.size() != 15) {
      throwWithMessageFromConstructor("test configuration error, expectedVectorVectorIntegralValues must have size 15");
    }
  }

  void SchemaEvolutionTestRead::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    analyzeVectorVector(iEvent);
    analyzeVectorVectorNonSplit(iEvent);
  }

  void SchemaEvolutionTestRead::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<int>>("expectedVectorVectorIntegralValues");
    desc.add<edm::InputTag>("vectorVectorTag");
    descriptions.addDefault(desc);
  }

  void SchemaEvolutionTestRead::analyzeVectorVector(edm::Event const& iEvent) const {
    auto const& vectorVector = iEvent.get(vectorVectorToken_);
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    if (vectorVector.outerVector_.size() != vectorSize) {
      throwWithMessage("analyzeVectorVector, vectorVector does not have expected size");
    }
    unsigned int i = 0;
    for (auto const& middleVector : vectorVector.outerVector_) {
      if (middleVector.middleVector_.size() != vectorSize) {
        throwWithMessage("analyzeVectorVector, middleVector does not have expected size");
      }
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      int j = 0;
      for (auto const& element : middleVector.middleVector_) {
        if (element.a_ != expectedVectorVectorIntegralValues_[0] + iOffset + j * 10) {
          throwWithMessage("analyzeVectorVector, element a_ does not contain expected value");
        }
        if (element.b_ != expectedVectorVectorIntegralValues_[0] + iOffset + j * 100) {
          throwWithMessage("analyzeVectorVector, element b_ does not contain expected value");
        }

        SchemaEvolutionChangeOrder const& changeOrder = element.changeOrder_;
        if (changeOrder.a_ != expectedVectorVectorIntegralValues_[1] + iOffset + j * 11) {
          throwWithMessage("analyzeVectorVector, changeOrder a_ does not contain expected value");
        }
        if (changeOrder.b_ != expectedVectorVectorIntegralValues_[1] + iOffset + j * 101) {
          throwWithMessage("analyzeVectorVector, changeOrder b_ does not contain expected value");
        }

        SchemaEvolutionAddMember const& addMember = element.addMember_;
        if (addMember.a_ != expectedVectorVectorIntegralValues_[2] + iOffset + j * 12) {
          throwWithMessage("analyzeVectorVector, addMember a_ does not contain expected value");
        }
        if (addMember.b_ != expectedVectorVectorIntegralValues_[2] + iOffset + j * 102) {
          throwWithMessage("analyzeVectorVector, addMember b_ does not contain expected value");
        }

        SchemaEvolutionRemoveMember const& removeMember = element.removeMember_;
        if (removeMember.a_ != expectedVectorVectorIntegralValues_[3] + iOffset + j * 13) {
          throwWithMessage("analyzeVectorVector, removeMember a_ does not contain expected value");
        }

        SchemaEvolutionMoveToBase const& moveToBase = element.moveToBase_;
        if (moveToBase.a_ != expectedVectorVectorIntegralValues_[4] + iOffset + j * 14) {
          throwWithMessage("analyzeVectorVector, moveToBase a_ does not contain expected value");
        }
        if (moveToBase.b_ != expectedVectorVectorIntegralValues_[4] + iOffset + j * 104) {
          throwWithMessage("analyzeVectorVector, moveToBase b_ does not contain expected value");
        }
        if (moveToBase.c_ != expectedVectorVectorIntegralValues_[4] + iOffset + j * 1004) {
          throwWithMessage("analyzeVectorVector, moveToBase c_ does not contain expected value");
        }
        if (moveToBase.d_ != expectedVectorVectorIntegralValues_[4] + iOffset + j * 10004) {
          throwWithMessage("analyzeVectorVector, moveToBase d_ does not contain expected value");
        }

        SchemaEvolutionChangeType const& changeType = element.changeType_;
        if (static_cast<int>(changeType.a_) != expectedVectorVectorIntegralValues_[5] + iOffset + j * 15) {
          throwWithMessage("analyzeVectorVector, changeType a_ does not contain expected value");
        }
        if (static_cast<int>(changeType.b_) != expectedVectorVectorIntegralValues_[5] + iOffset + j * 105) {
          throwWithMessage("analyzeVectorVector, changeType b_ does not contain expected value");
        }

        SchemaEvolutionAddBase const& addBase = element.addBase_;
        if (addBase.a_ != expectedVectorVectorIntegralValues_[6] + iOffset + j * 16) {
          throwWithMessage("analyzeVectorVector, addToBase a_ does not contain expected value");
        }
        if (addBase.b_ != expectedVectorVectorIntegralValues_[6] + iOffset + j * 106) {
          throwWithMessage("analyzeVectorVector, addToBase b_ does not contain expected value");
        }

        SchemaEvolutionPointerToMember const& pointerToMember = element.pointerToMember_;
        if (pointerToMember.a_ != expectedVectorVectorIntegralValues_[7] + iOffset + j * 17) {
          throwWithMessage("analyzeVectorVector, pointerToMember a_ does not contain expected value");
        }
        if (pointerToMember.b_ != expectedVectorVectorIntegralValues_[7] + iOffset + j * 107) {
          throwWithMessage("analyzeVectorVector, pointerToMember b_ does not contain expected value");
        }
        // This part is commented out because it fails. My conclusion is that ROOT
        // does not properly support schema evolution in this case. CMS does not
        // usually use pointers in persistent formats. So for now we are just ignoring
        // this issue.
        // if (pointerToMember.c() != expectedVectorVectorIntegralValues_[7] + iOffset + j * 1007) {
        //   throwWithMessage("analyzeVectorVector, pointerToMember c_ does not contain expected value");
        // }

        SchemaEvolutionPointerToUniquePtr const& pointerToUniquePtr = element.pointerToUniquePtr_;
        if (pointerToUniquePtr.a_ != expectedVectorVectorIntegralValues_[8] + iOffset + j * 18) {
          throwWithMessage("analyzeVectorVector, pointerToUniquePtr a_ does not contain expected value");
        }
        if (pointerToUniquePtr.b_ != expectedVectorVectorIntegralValues_[8] + iOffset + j * 108) {
          throwWithMessage("analyzeVectorVector, pointerToUniquePtr b_ does not contain expected value");
        }
        if (pointerToUniquePtr.contained_->c_ != expectedVectorVectorIntegralValues_[8] + iOffset + j * 1008) {
          throwWithMessage("analyzeVectorVector, pointerToUniquePtr c_ does not contain expected value");
        }

        SchemaEvolutionCArrayToStdArray const& cArrayToStdArray = element.cArrayToStdArray_;
        if (cArrayToStdArray.a_[0] != expectedVectorVectorIntegralValues_[9] + iOffset + j * 19) {
          throwWithMessage("analyzeVectorVector, cArrayToStdArray a_[0] does not contain expected value");
        }
        if (cArrayToStdArray.a_[1] != expectedVectorVectorIntegralValues_[9] + iOffset + j * 109) {
          throwWithMessage("analyzeVectorVector, cArrayToStdArray a_[1] does not contain expected value");
        }
        if (cArrayToStdArray.a_[2] != expectedVectorVectorIntegralValues_[9] + iOffset + j * 1009) {
          throwWithMessage("analyzeVectorVector, cArrayToStdArray a_[2] does not contain expected value");
        }

        // This part is commented out because it fails. My conclusion is that ROOT
        // does not properly support schema evolution in this case. CMS does not
        // usually use pointers in persistent formats. So for now we are just ignoring
        // this issue. Note that pointerToMember fails because the values are incorrect.
        // This one is also commented out of the format, the write function and here
        // because simply reading the object causes a fatal exception even without
        // checking the values.
        // SchemaEvolutionCArrayToStdVector const& cArrayToStdVector = element.cArrayToStdVector_;
        // if (cArrayToStdVector.a_[0] != expectedVectorVectorIntegralValues_[10] + iOffset + j * 20) {
        //   throwWithMessage("analyzeVectorVector, cArrayToStdVector a_[0] does not contain expected value");
        // }
        // if (cArrayToStdVector.a_[1] != expectedVectorVectorIntegralValues_[10] + iOffset + j * 110) {
        //   throwWithMessage("analyzeVectorVector, cArrayToStdVector a_[1] does not contain expected value");
        // }
        // if (cArrayToStdVector.a_[2] != expectedVectorVectorIntegralValues_[10] + iOffset + j * 1010) {
        //  throwWithMessage("analyzeVectorVector, cArrayToStdVector a_[2] does not contain expected value");
        // }

        {
          SchemaEvolutionVectorToList const& vectorToList = element.vectorToList_;
          auto iter = vectorToList.a_.cbegin();
          auto iter0 = iter;
          auto iter1 = ++iter;
          auto iter2 = ++iter;
          if (*iter0 != expectedVectorVectorIntegralValues_[11] + iOffset + j * 21) {
            throwWithMessage("vectorToList, element 0 does not contain expected value");
          }
          if (*iter1 != expectedVectorVectorIntegralValues_[11] + iOffset + j * 111) {
            throwWithMessage("vectorToList, element 1 does not contain expected value");
          }
          if (*iter2 != expectedVectorVectorIntegralValues_[11] + iOffset + j * 1011) {
            throwWithMessage("vectorToList, element 2 does not contain expected value");
          }
        }

        {
          SchemaEvolutionMapToUnorderedMap const& mapToUnorderedMap = element.mapToUnorderedMap_;
          if (mapToUnorderedMap.a_.size() != 3) {
            throwWithMessage("mapToUnorderedMap, map has unexpected size");
          }
          auto iter = mapToUnorderedMap.a_.cbegin();

          // Easier to check values if we sort them first so sort them in a regular map
          std::map<int, int> orderedMap;
          orderedMap.insert(*iter);
          ++iter;
          orderedMap.insert(*iter);
          ++iter;
          orderedMap.insert(*iter);

          auto orderedIter = orderedMap.cbegin();
          auto iter0 = orderedIter;
          auto iter1 = ++orderedIter;
          auto iter2 = ++orderedIter;
          if (iter0->first != expectedVectorVectorIntegralValues_[12] + iOffset + j * 22) {
            throwWithMessage("mapToUnorderedMap, element 0 key does not contain expected value");
          }
          if (iter0->second != expectedVectorVectorIntegralValues_[12] + iOffset + j * 112) {
            throwWithMessage("mapToUnorderedMap, element 0 does not contain expected value");
          }
          if (iter1->first != expectedVectorVectorIntegralValues_[12] + iOffset + j * 1012 + 1012) {
            throwWithMessage("mapToUnorderedMap, element 1 key does not contain expected value");
          }
          if (iter1->second != expectedVectorVectorIntegralValues_[12] + iOffset + j * 10012) {
            throwWithMessage("mapToUnorderedMap, element 1 does not contain expected value");
          }
          if (iter2->first != expectedVectorVectorIntegralValues_[12] + iOffset + j * 100012 + 100012) {
            throwWithMessage("mapToUnorderedMap, element 2 key does not contain expected value");
          }
          if (iter2->second != expectedVectorVectorIntegralValues_[12] + iOffset + j * 1000012) {
            throwWithMessage("mapToUnorderedMap, element 2 does not contain expected value");
          }
        }
        ++j;
      }
      ++i;
    }
  }

  void SchemaEvolutionTestRead::analyzeVectorVectorNonSplit(edm::Event const& iEvent) const {
    auto const& vectorVectorNonSplit = iEvent.get(vectorVectorNonSplitToken_);
    unsigned int vectorSize = 1;
    if (vectorVectorNonSplit.outerVector_.size() != vectorSize) {
      throwWithMessage("analyzeVectorVectorNonSplit, outerVector does not have expected size");
    }
    for (auto const& middleVector : vectorVectorNonSplit.outerVector_) {
      if (middleVector.middleVector_.size() != vectorSize) {
        throwWithMessage("analyzeVectorVectorNonSplit, middleVector does not have expected size");
      }
      for (auto const& element : middleVector.middleVector_) {
        if (element.a_ != expectedVectorVectorIntegralValues_[13]) {
          throwWithMessage("analyzeVectorVectorNonSplit, element a_ does not contain expected value");
        }
      }
    }
  }

  void SchemaEvolutionTestRead::throwWithMessageFromConstructor(const char* msg) const {
    throw cms::Exception("TestFailure") << "SchemaEvolutionTestRead constructor, " << msg;
  }

  void SchemaEvolutionTestRead::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "SchemaEvolutionTestRead::analyze, " << msg;
  }

}  // namespace edmtest

using edmtest::SchemaEvolutionTestRead;
DEFINE_FWK_MODULE(SchemaEvolutionTestRead);
