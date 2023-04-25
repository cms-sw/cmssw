#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/TrackOfDSVThings.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include <vector>

namespace edmtest {

  class ThinningDSVTestAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit ThinningDSVTestAnalyzer(edm::ParameterSet const& pset);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& c) const override;

  private:
    void incrementExpectedValue(std::vector<int>::const_iterator& iter) const;

    edm::EDGetTokenT<edmNew::DetSetVector<Thing>> parentToken_;
    edm::EDGetTokenT<edmNew::DetSetVector<Thing>> thinnedToken_;
    edm::EDGetTokenT<edm::ThinnedAssociation> associationToken_;
    edm::EDGetTokenT<TrackOfDSVThingsCollection> trackToken_;

    struct DSContent {
      unsigned int id;
      std::vector<int> values;
    };

    bool parentWasDropped_;
    std::vector<DSContent> expectedParentContent_;
    bool thinnedWasDropped_;
    bool thinnedIsAlias_;
    bool refToParentIsAvailable_;
    std::vector<DSContent> expectedThinnedContent_;
    std::vector<unsigned int> expectedIndexesIntoParent_;
    bool associationShouldBeDropped_;
    unsigned int expectedNumberOfTracks_;
    std::vector<int> expectedValues_;
    int parentSlimmedValueFactor_;
    int thinnedSlimmedValueFactor_;
    int refSlimmedValueFactor_;
  };

  ThinningDSVTestAnalyzer::ThinningDSVTestAnalyzer(edm::ParameterSet const& pset)
      : parentToken_(consumes<edmNew::DetSetVector<Thing>>(pset.getParameter<edm::InputTag>("parentTag"))),
        thinnedToken_(consumes<edmNew::DetSetVector<Thing>>(pset.getParameter<edm::InputTag>("thinnedTag"))),
        associationToken_(consumes<edm::ThinnedAssociation>(pset.getParameter<edm::InputTag>("associationTag"))),
        trackToken_(consumes<TrackOfDSVThingsCollection>(pset.getParameter<edm::InputTag>("trackTag"))),
        parentWasDropped_(pset.getParameter<bool>("parentWasDropped")),
        thinnedWasDropped_(pset.getParameter<bool>("thinnedWasDropped")),
        thinnedIsAlias_(pset.getParameter<bool>("thinnedIsAlias")),
        refToParentIsAvailable_(pset.getParameter<bool>("refToParentIsAvailable")),
        associationShouldBeDropped_(pset.getParameter<bool>("associationShouldBeDropped")),
        expectedNumberOfTracks_(pset.getParameter<unsigned int>("expectedNumberOfTracks")),
        expectedValues_(pset.getParameter<std::vector<int>>("expectedValues")) {
    auto makeDSContent = [](edm::ParameterSet const& p) {
      return DSContent{p.getParameter<unsigned int>("id"), p.getParameter<std::vector<int>>("values")};
    };
    if (!parentWasDropped_) {
      expectedParentContent_ = edm::vector_transform(
          pset.getParameter<std::vector<edm::ParameterSet>>("expectedParentContent"), makeDSContent);
    }
    if (!thinnedWasDropped_) {
      expectedThinnedContent_ = edm::vector_transform(
          pset.getParameter<std::vector<edm::ParameterSet>>("expectedThinnedContent"), makeDSContent);
    }
    if (!associationShouldBeDropped_) {
      expectedIndexesIntoParent_ = pset.getParameter<std::vector<unsigned int>>("expectedIndexesIntoParent");
    }

    auto slimmedFactor = [](int count, int factor) {
      int ret = 1;
      for (int i = 0; i < count; ++i) {
        ret *= factor;
      }
      return ret;
    };
    int const slimmedValueFactor = pset.getParameter<int>("slimmedValueFactor");
    parentSlimmedValueFactor_ = slimmedFactor(pset.getParameter<int>("parentSlimmedCount"), slimmedValueFactor);
    thinnedSlimmedValueFactor_ = slimmedFactor(pset.getParameter<int>("thinnedSlimmedCount"), slimmedValueFactor);
    refSlimmedValueFactor_ = slimmedFactor(pset.getParameter<int>("refSlimmedCount"), slimmedValueFactor);
  }

  void ThinningDSVTestAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("parentTag");
    desc.add<edm::InputTag>("thinnedTag");
    desc.add<edm::InputTag>("associationTag");
    desc.add<edm::InputTag>("trackTag");
    desc.add<bool>("parentWasDropped", false);
    desc.add<bool>("thinnedWasDropped", false);
    desc.add<bool>("thinnedIsAlias", false);
    desc.add<bool>("refToParentIsAvailable", true)
        ->setComment(
            "If Ref-to-parent is generally available. With thinnedRefFrom it may happen that the Ref-to-parent is not "
            "available, but the Ref-to-thinned is. In such case this parameter should be set to 'False', and the "
            "'expectedValues' should be set to correspond the values via Ref-to-thinned.");
    std::vector<int> defaultV;
    std::vector<unsigned int> defaultVU;
    std::vector<edm::ParameterSet> defaultVPSet;
    edm::ParameterSetDescription dsValidator;
    dsValidator.add<unsigned int>("id");
    dsValidator.add<std::vector<int>>("values", defaultV);
    desc.addVPSet("expectedParentContent", dsValidator, defaultVPSet);
    desc.addVPSet("expectedThinnedContent", dsValidator, defaultVPSet);
    desc.add<std::vector<unsigned int>>("expectedIndexesIntoParent", defaultVU);
    desc.add<bool>("associationShouldBeDropped", false);
    desc.add<unsigned int>("expectedNumberOfTracks", 5);
    desc.add<std::vector<int>>("expectedValues");
    desc.add<int>("parentSlimmedCount", 0);
    desc.add<int>("thinnedSlimmedCount", 0);
    desc.add<int>("refSlimmedCount", 0);
    desc.add<int>("slimmedValueFactor", 10);
    descriptions.addDefault(desc);
  }

  void ThinningDSVTestAnalyzer::analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const {
    auto parentCollection = event.getHandle(parentToken_);
    auto thinnedCollection = event.getHandle(thinnedToken_);
    auto associationCollection = event.getHandle(associationToken_);
    auto trackCollection = event.getHandle(trackToken_);

    if (parentWasDropped_) {
      if (parentCollection.isValid()) {
        throw cms::Exception("TestFailure") << "parent collection present, should have been dropped";
      }
    } else if (!expectedParentContent_.empty()) {
      if (parentCollection->size() != expectedParentContent_.size()) {
        throw cms::Exception("TestFailure") << "parent collection has unexpected size, got " << parentCollection->size()
                                            << " expected " << expectedParentContent_.size();
      }

      auto iExpected = expectedParentContent_.begin();
      for (auto const& detset : *parentCollection) {
        auto const& expectedContent = *iExpected;
        ++iExpected;

        if (detset.id() != expectedContent.id) {
          throw cms::Exception("TestFailure")
              << "parent collection detset has unexpected id " << detset.id() << " expected " << expectedContent.id;
        }
        if (detset.size() != expectedContent.values.size()) {
          throw cms::Exception("TestFailure")
              << "parent collection detset with id " << detset.id() << " has unexpected size, got " << detset.size()
              << " expected " << expectedContent.values.size();
        }
        auto iValue = expectedContent.values.begin();
        for (auto const& thing : detset) {
          // Just some numbers that match the somewhat arbitrary values put in
          // by the ThingProducer.
          int expected =
              static_cast<int>(*iValue + (event.eventAuxiliary().event() - 1) * 100) * parentSlimmedValueFactor_;
          if (thing.a != expected) {
            throw cms::Exception("TestFailure")
                << "parent collection has unexpected content for detset with id " << detset.id() << ", got " << thing.a
                << " expected " << expected << " (element " << std::distance(expectedContent.values.begin(), iValue)
                << ")";
          }
          ++iValue;
        }
      }
    }

    // Check to see the content is what we expect based on what was written
    // by ThingProducer and TrackOfThingsProducer. The values are somewhat
    // arbitrary and meaningless.
    edm::RefProd<edmNew::DetSetVector<Thing>> thinnedRefProd;
    if (thinnedWasDropped_) {
      if (thinnedCollection.isValid()) {
        throw cms::Exception("TestFailure") << "thinned collection present, should have been dropped";
      }
    } else {
      thinnedRefProd = edm::RefProd<edmNew::DetSetVector<Thing>>{thinnedCollection};
      if (thinnedCollection->size() != expectedThinnedContent_.size()) {
        throw cms::Exception("TestFailure")
            << "thinned collection has unexpected size, got " << thinnedCollection->size() << " expected "
            << expectedThinnedContent_.size();
      }

      auto iExpected = expectedThinnedContent_.begin();
      for (auto const& detset : *thinnedCollection) {
        auto const& expectedContent = *iExpected;
        ++iExpected;

        if (detset.id() != expectedContent.id) {
          throw cms::Exception("TestFailure")
              << "thinned collection detset has unexpected id " << detset.id() << " expected " << expectedContent.id;
        }
        if (detset.size() != expectedContent.values.size()) {
          throw cms::Exception("TestFailure")
              << "thinned collection detset with id " << detset.id() << " has unexpected size, got " << detset.size()
              << " expected " << expectedContent.values.size();
        }
        auto iValue = expectedContent.values.begin();
        for (auto const& thing : detset) {
          int expected =
              static_cast<int>(*iValue + (event.eventAuxiliary().event() - 1) * 100) * thinnedSlimmedValueFactor_;
          if (thing.a != expected) {
            throw cms::Exception("TestFailure")
                << "thinned collection has unexpected content for detset with id " << detset.id() << ", got " << thing.a
                << " expected " << expected << " (element " << std::distance(expectedContent.values.begin(), iValue)
                << ")";
          }
          ++iValue;
        }
      }
    }

    if (associationShouldBeDropped_ && associationCollection.isValid()) {
      throw cms::Exception("TestFailure") << "association collection should have been dropped but was not";
    }
    if (!associationShouldBeDropped_) {
      unsigned int expectedIndex = 0;
      if (associationCollection->indexesIntoParent().size() != expectedIndexesIntoParent_.size()) {
        throw cms::Exception("TestFailure")
            << "association collection has unexpected size " << associationCollection->indexesIntoParent().size()
            << " expected " << expectedIndexesIntoParent_.size();
      }
      for (auto const& association : associationCollection->indexesIntoParent()) {
        if (association != expectedIndexesIntoParent_.at(expectedIndex)) {
          throw cms::Exception("TestFailure")
              << "association collection has unexpected content, for index " << expectedIndex << " got " << association
              << " expected " << expectedIndexesIntoParent_.at(expectedIndex);
        }
        ++expectedIndex;
      }
    }

    if (!parentWasDropped_ && !associationShouldBeDropped_) {
      if (associationCollection->parentCollectionID() != parentCollection.id()) {
        throw cms::Exception("TestFailure") << "analyze parent ProductID is not correct";
      }
    }

    if (!thinnedWasDropped_ && !associationShouldBeDropped_) {
      if (associationCollection->thinnedCollectionID() != thinnedCollection.id()) {
        throw cms::Exception("TestFailure") << "analyze thinned ProductID is not correct";
      }
    }

    if (trackCollection->size() != expectedNumberOfTracks_) {
      throw cms::Exception("TestFailure")
          << "unexpected Track size " << trackCollection->size() << " expected " << expectedNumberOfTracks_;
    }

    if (expectedValues_.empty()) {
      return;
    }

    int eventOffset = (static_cast<int>(event.eventAuxiliary().event()) - 1) * 100;

    std::vector<int>::const_iterator expectedValue = expectedValues_.begin();
    for (auto const& track : *trackCollection) {
      if (not refToParentIsAvailable_ or *expectedValue == -1) {
        if (track.ref1.isAvailable()) {
          throw cms::Exception("TestFailure") << "ref1 is available when it should not be, refers to "
                                              << track.ref1.id() << " key " << track.ref1.key();
        }
      } else {
        if (!track.ref1.isAvailable()) {
          throw cms::Exception("TestFailure") << "ref1 is not available when it should be";
        }
        // Check twice to test some possible caching problems.
        const int expected = (*expectedValue + eventOffset) * refSlimmedValueFactor_;
        if (track.ref1->a != expected) {
          throw cms::Exception("TestFailure")
              << "Unexpected values from ref1, got " << track.ref1->a << " expected " << expected;
        }
        if (track.ref1->a != expected) {
          throw cms::Exception("TestFailure")
              << "Unexpected values from ref1 (2nd try), got " << track.ref1->a << " expected " << expected;
          ;
        }
      }

      if (not thinnedWasDropped_) {
        auto refToThinned = edm::thinnedRefFrom(track.ref1, thinnedRefProd, event.productGetter());
        if (*expectedValue == -1) {
          if (refToThinned.isNonnull()) {
            throw cms::Exception("TestFailure") << "thinnedRefFrom(ref1) is non-null when it should be null";
          }
          if (refToThinned.isAvailable()) {
            throw cms::Exception("TestFailure") << "thinnedRefFrom(ref1) is available when it should not be";
          }
        } else {
          if (refToThinned.isNull()) {
            throw cms::Exception("TestFailure") << "thinnedRefFrom(ref1) is null when it should not be";
          }
          if (refToThinned.id() != thinnedCollection.id()) {
            throw cms::Exception("TestFailure") << "thinnedRefFrom(ref).id() " << refToThinned.id()
                                                << " differs from expectation " << thinnedCollection.id();
          }
          if (not refToThinned.isAvailable()) {
            throw cms::Exception("TestFailure") << "thinnedRefFrom(ref1) is not available when it should be";
          }
          // Check twice to test some possible caching problems.
          // need to account for slimming because going through an explicit ref-to-slimmed
          const int expected = (*expectedValue + eventOffset) * thinnedSlimmedValueFactor_;
          if (refToThinned->a != expected) {
            throw cms::Exception("TestFailure")
                << "Unexpected values from thinnedRefFrom(ref1), got " << refToThinned->a << " expected " << expected;
          }
          if (refToThinned->a != expected) {
            throw cms::Exception("TestFailure") << "Unexpected values from thinnedRefFrom(ref1) (2nd try) "
                                                << refToThinned->a << " expected " << expected;
          }
        }
      }
      incrementExpectedValue(expectedValue);

      if (not refToParentIsAvailable_ or *expectedValue == -1) {
        if (track.ref2.isAvailable()) {
          throw cms::Exception("TestFailure") << "ref2 is available when it should not be";
        }
      } else {
        if (!track.ref2.isAvailable()) {
          throw cms::Exception("TestFailure") << "ref2 is not available when it should be";
        }

        const int expected = (*expectedValue + eventOffset) * refSlimmedValueFactor_;
        if (track.ref2->a != expected) {
          throw cms::Exception("TestFailure")
              << "unexpected values from ref2, got " << track.ref2->a << " expected " << expected;
        }
        if (track.ref2->a != expected) {
          throw cms::Exception("TestFailure")
              << "unexpected values from ref2 (2nd try), got " << track.ref2->a << " expected " << expected;
        }
      }

      if (not thinnedWasDropped_) {
        auto refToThinned = edm::thinnedRefFrom(track.ref2, thinnedRefProd, event.productGetter());
        if (*expectedValue == -1) {
          if (refToThinned.isNonnull()) {
            throw cms::Exception("TestFailure") << "thinnedRefFrom(ref2) is non-null when it should be null";
          }
          if (refToThinned.isAvailable()) {
            throw cms::Exception("TestFailure") << "thinnedRefFrom(ref2) is available when it should not be";
          }
        } else {
          if (refToThinned.isNull()) {
            throw cms::Exception("TestFailure") << "thinnedRefFrom(ref2) is null when it should not be";
          }
          if (refToThinned.id() != thinnedCollection.id()) {
            throw cms::Exception("TestFailure") << "thinnedRefFrom(ref2).id() " << refToThinned.id()
                                                << " differs from expectation " << thinnedCollection.id();
          }
          if (not refToThinned.isAvailable()) {
            throw cms::Exception("TestFailure") << "thinnedRefFrom(ref2) is not available when it should be";
          }
          // Check twice to test some possible caching problems.
          // need to account for slimming because going through an explicit ref-to-slimmed
          const int expected = (*expectedValue + eventOffset) * thinnedSlimmedValueFactor_;
          if (refToThinned->a != expected) {
            throw cms::Exception("TestFailure")
                << "Unexpected values from thinnedRefFrom(ref2), got " << refToThinned->a << " expected " << expected;
          }
          if (refToThinned->a != expected) {
            throw cms::Exception("TestFailure") << "Unexpected values from thinnedRefFrom(ref2) (2nd try), got "
                                                << refToThinned->a << " expected " << expected;
          }
        }
      }

      incrementExpectedValue(expectedValue);

      // Test RefVector
      unsigned int k = 0;
      bool allPresent = true;
      for (auto iExpectedValue : expectedValues_) {
        if (iExpectedValue != -1) {
          if (not thinnedWasDropped_) {
            auto refToThinned = edm::thinnedRefFrom(track.refVector1[k], thinnedRefProd, event.productGetter());
            // need to account for slimming because going through an explicit ref-to-slimmed
            const int expected = (iExpectedValue + eventOffset) * thinnedSlimmedValueFactor_;
            if (refToThinned->a != expected) {
              throw cms::Exception("TestFailure") << "unexpected values from thinnedRefFrom(refVector1), got "
                                                  << refToThinned->a << " expected " << expected;
            }
          }
          if (refToParentIsAvailable_) {
            const int expected = (iExpectedValue + eventOffset) * refSlimmedValueFactor_;
            if (track.refVector1[k]->a != expected) {
              throw cms::Exception("TestFailure")
                  << "unexpected values from refVector1, got " << track.refVector1[k]->a << " expected " << expected;
            }
          }
        } else {
          allPresent = false;
        }
        ++k;
      }

      if (refToParentIsAvailable_ and allPresent) {
        if (!track.refVector1.isAvailable()) {
          throw cms::Exception("TestFailure") << "unexpected value (false) from refVector::isAvailable";
        }
      } else {
        if (track.refVector1.isAvailable()) {
          throw cms::Exception("TestFailure") << "unexpected value (true) from refVector::isAvailable";
        }
      }
      k = 0;
      for (auto iExpectedValue : expectedValues_) {
        if (refToParentIsAvailable_ and iExpectedValue != -1) {
          const int expected = (iExpectedValue + eventOffset) * refSlimmedValueFactor_;
          if (track.refVector1[k]->a != expected) {
            throw cms::Exception("TestFailure")
                << "unexpected values from refVector1, got " << track.refVector1[k]->a << " expected " << expected;
          }
        } else {
          allPresent = false;
        }
        ++k;
      }
    }
  }

  void ThinningDSVTestAnalyzer::incrementExpectedValue(std::vector<int>::const_iterator& iter) const {
    ++iter;
    if (iter == expectedValues_.end())
      iter = expectedValues_.begin();
  }
}  // namespace edmtest
using edmtest::ThinningDSVTestAnalyzer;
DEFINE_FWK_MODULE(ThinningDSVTestAnalyzer);
