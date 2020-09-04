#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/TrackOfThings.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include <vector>

namespace edmtest {

  class ThinningTestAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit ThinningTestAnalyzer(edm::ParameterSet const& pset);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void analyze(edm::StreamID, edm::Event const& e, edm::EventSetup const& c) const override;

  private:
    edm::Handle<ThingCollection> getParent(edm::Event const& event) const;
    std::tuple<edm::Handle<ThingCollection>, edm::RefProd<ThingCollection>> getThinned(edm::Event const& event) const;
    edm::Handle<edm::ThinnedAssociation> getAssociation(edm::Event const& event) const;
    edm::Handle<TrackOfThingsCollection> getTrackCollection(edm::Event const& event) const;

    template <typename RefT>
    void testRefToParent(RefT const& ref,
                         std::string_view refName,
                         int const expectedValue,
                         int const eventOffset) const;

    template <typename RefT>
    void testRefToThinned(RefT const& ref,
                          std::string_view refName,
                          int const expectedValue,
                          int const eventOffset,
                          edm::ProductID const thinnedCollectionID) const;

    void testVectors(TrackOfThings const& track,
                     edm::RefProd<ThingCollection> const& thinnedRefProd,
                     edm::EDProductGetter const& productGetter,
                     int const eventOffset) const;

    void incrementExpectedValue(std::vector<int>::const_iterator& iter) const;

    edm::EDGetTokenT<ThingCollection> parentToken_;
    edm::EDGetTokenT<ThingCollection> thinnedToken_;
    edm::EDGetTokenT<edm::ThinnedAssociation> associationToken_;
    edm::EDGetTokenT<TrackOfThingsCollection> trackToken_;

    bool parentWasDropped_;
    std::vector<int> expectedParentContent_;
    bool thinnedWasDropped_;
    bool thinnedIsAlias_;
    bool refToParentIsAvailable_;
    std::vector<int> expectedThinnedContent_;
    std::vector<unsigned int> expectedIndexesIntoParent_;
    bool associationShouldBeDropped_;
    std::vector<int> expectedValues_;
    int parentSlimmedValueFactor_;
    int thinnedSlimmedValueFactor_;
    int refSlimmedValueFactor_;
  };

  ThinningTestAnalyzer::ThinningTestAnalyzer(edm::ParameterSet const& pset) {
    parentToken_ = consumes<ThingCollection>(pset.getParameter<edm::InputTag>("parentTag"));
    thinnedToken_ = mayConsume<ThingCollection>(pset.getParameter<edm::InputTag>("thinnedTag"));
    associationToken_ = mayConsume<edm::ThinnedAssociation>(pset.getParameter<edm::InputTag>("associationTag"));
    trackToken_ = consumes<TrackOfThingsCollection>(pset.getParameter<edm::InputTag>("trackTag"));
    parentWasDropped_ = pset.getParameter<bool>("parentWasDropped");
    if (!parentWasDropped_) {
      expectedParentContent_ = pset.getParameter<std::vector<int>>("expectedParentContent");
    }
    thinnedWasDropped_ = pset.getParameter<bool>("thinnedWasDropped");
    thinnedIsAlias_ = pset.getParameter<bool>("thinnedIsAlias");
    if (!thinnedWasDropped_) {
      expectedThinnedContent_ = pset.getParameter<std::vector<int>>("expectedThinnedContent");
    }
    refToParentIsAvailable_ = pset.getParameter<bool>("refToParentIsAvailable");
    associationShouldBeDropped_ = pset.getParameter<bool>("associationShouldBeDropped");
    if (!associationShouldBeDropped_) {
      expectedIndexesIntoParent_ = pset.getParameter<std::vector<unsigned int>>("expectedIndexesIntoParent");
    }
    expectedValues_ = pset.getParameter<std::vector<int>>("expectedValues");

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

  void ThinningTestAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
    desc.add<std::vector<int>>("expectedParentContent", defaultV);
    desc.add<std::vector<int>>("expectedThinnedContent", defaultV);
    desc.add<std::vector<unsigned int>>("expectedIndexesIntoParent", defaultVU);
    desc.add<bool>("associationShouldBeDropped", false);
    desc.add<std::vector<int>>("expectedValues");
    desc.add<int>("parentSlimmedCount", 0);
    desc.add<int>("thinnedSlimmedCount", 0);
    desc.add<int>("refSlimmedCount", 0);
    desc.add<int>("slimmedValueFactor", 10);
    descriptions.addDefault(desc);
  }

  void ThinningTestAnalyzer::analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const {
    auto parentCollection = getParent(event);
    auto [thinnedCollection, thinnedRefProd] = getThinned(event);
    auto associationCollection = getAssociation(event);
    auto trackCollection = getTrackCollection(event);

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

    int const eventOffset = static_cast<int>(event.eventAuxiliary().event() * 100 + 100);

    std::vector<int>::const_iterator expectedValue = expectedValues_.begin();
    for (auto const& track : *trackCollection) {
      testRefToParent(track.ref1, "ref1", *expectedValue, eventOffset);
      if (not thinnedWasDropped_) {
        testRefToThinned(edm::thinnedRefFrom(track.ref1, thinnedRefProd, event.productGetter()),
                         "ref1",
                         *expectedValue,
                         eventOffset,
                         thinnedCollection.id());
      }
      testRefToParent(track.refToBase1, "refToBase1", *expectedValue, eventOffset);
      incrementExpectedValue(expectedValue);

      testRefToParent(track.ref2, "ref2", *expectedValue, eventOffset);
      if (not thinnedWasDropped_) {
        testRefToThinned(edm::thinnedRefFrom(track.ref2, thinnedRefProd, event.productGetter()),
                         "ref2",
                         *expectedValue,
                         eventOffset,
                         thinnedCollection.id());
      }
      incrementExpectedValue(expectedValue);

      testRefToParent(track.ptr1, "ptr1", *expectedValue, eventOffset);
      incrementExpectedValue(expectedValue);

      testRefToParent(track.ptr2, "ptr2", *expectedValue, eventOffset);
      incrementExpectedValue(expectedValue);

      testVectors(track, thinnedRefProd, event.productGetter(), eventOffset);
    }
  }

  edm::Handle<ThingCollection> ThinningTestAnalyzer::getParent(edm::Event const& event) const {
    auto parentCollection = event.getHandle(parentToken_);

    unsigned int i = 0;
    if (parentWasDropped_) {
      if (parentCollection.isValid()) {
        throw cms::Exception("TestFailure") << "parent collection present, should have been dropped";
      }
    } else if (!expectedParentContent_.empty()) {
      if (parentCollection->size() != expectedParentContent_.size()) {
        throw cms::Exception("TestFailure") << "parent collection has unexpected size, got " << parentCollection->size()
                                            << " expected " << expectedParentContent_.size();
      }
      for (auto const& thing : *parentCollection) {
        // Just some numbers that match the somewhat arbitrary values put in
        // by the ThingProducer.
        int expected = static_cast<int>(expectedParentContent_.at(i) + event.eventAuxiliary().event() * 100 + 100) *
                       parentSlimmedValueFactor_;
        if (thing.a != expected) {
          throw cms::Exception("TestFailure")
              << "parent collection has unexpected content, got " << thing.a << " expected " << expected;
        }
        ++i;
      }
    }

    return parentCollection;
  }

  std::tuple<edm::Handle<ThingCollection>, edm::RefProd<ThingCollection>> ThinningTestAnalyzer::getThinned(
      edm::Event const& event) const {
    auto thinnedCollection = event.getHandle(thinnedToken_);

    // Check to see the content is what we expect based on what was written
    // by ThingProducer and TrackOfThingsProducer. The values are somewhat
    // arbitrary and meaningless.
    edm::RefProd<ThingCollection> thinnedRefProd;
    if (thinnedWasDropped_) {
      if (thinnedCollection.isValid()) {
        throw cms::Exception("TestFailure") << "thinned collection present, should have been dropped";
      }
    } else {
      thinnedRefProd = edm::RefProd<ThingCollection>{thinnedCollection};
      unsigned expectedIndex = 0;
      if (thinnedCollection->size() != expectedThinnedContent_.size()) {
        throw cms::Exception("TestFailure")
            << "thinned collection has unexpected size, got " << thinnedCollection->size() << " expected "
            << expectedThinnedContent_.size();
      }
      for (auto const& thing : *thinnedCollection) {
        const int expected = (expectedThinnedContent_.at(expectedIndex) + event.eventAuxiliary().event() * 100 + 100) *
                             thinnedSlimmedValueFactor_;
        if (thing.a != expected) {
          throw cms::Exception("TestFailure")
              << "thinned collection has unexpected content, got " << thing.a << " expected " << expected;
        }
        ++expectedIndex;
      }
    }
    return std::tuple(thinnedCollection, thinnedRefProd);
  }

  edm::Handle<edm::ThinnedAssociation> ThinningTestAnalyzer::getAssociation(edm::Event const& event) const {
    auto associationCollection = event.getHandle(associationToken_);
    if (associationShouldBeDropped_ && associationCollection.isValid()) {
      throw cms::Exception("TestFailure") << "association collection should have been dropped but was not";
    }
    if (!associationShouldBeDropped_) {
      unsigned int expectedIndex = 0;
      if (associationCollection->indexesIntoParent().size() != expectedIndexesIntoParent_.size()) {
        throw cms::Exception("TestFailure") << "association collection has unexpected size";
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
    return associationCollection;
  }

  edm::Handle<TrackOfThingsCollection> ThinningTestAnalyzer::getTrackCollection(edm::Event const& event) const {
    auto trackCollection = event.getHandle(trackToken_);
    if (trackCollection->size() != 5u) {
      throw cms::Exception("TestFailure") << "unexpected Track size";
    }
    return trackCollection;
  }

  template <typename RefT>
  void ThinningTestAnalyzer::testRefToParent(RefT const& ref,
                                             std::string_view refName,
                                             int const expectedValue,
                                             int const eventOffset) const {
    if (not refToParentIsAvailable_ or expectedValue == -1) {
      if (ref.isAvailable()) {
        throw cms::Exception("TestFailure")
            << refName << " is available when it should not be, refers to " << ref.id() << " key " << ref.key();
      }
    } else {
      if (!ref.isAvailable()) {
        throw cms::Exception("TestFailure") << refName << " is not available when it should be";
      }
      // Check twice to test some possible caching problems.
      const int expected = (expectedValue + eventOffset) * refSlimmedValueFactor_;
      if (ref->a != expected) {
        throw cms::Exception("TestFailure")
            << "Unexpected values from " << refName << ", got " << ref->a << " expected " << expected;
      }
      if (ref->a != expected) {
        throw cms::Exception("TestFailure")
            << "Unexpected values from " << refName << "  (2nd try), got " << ref->a << " expected " << expected;
      }
    }
  }

  template <typename RefT>
  void ThinningTestAnalyzer::testRefToThinned(RefT const& refToThinned,
                                              std::string_view refName,
                                              int const expectedValue,
                                              int const eventOffset,
                                              edm::ProductID const thinnedCollectionID) const {
    if (expectedValue == -1) {
      if (refToThinned.isNonnull()) {
        throw cms::Exception("TestFailure") << "thinnedRefFrom(" << refName << ") is non-null when it should be null";
      }
      if (refToThinned.isAvailable()) {
        throw cms::Exception("TestFailure") << "thinnedRefFrom(" << refName << ") is available when it should not be";
      }
    } else {
      if (refToThinned.isNull()) {
        throw cms::Exception("TestFailure") << "thinnedRefFrom(" << refName << ") is null when it should not be";
      }
      if (refToThinned.id() != thinnedCollectionID) {
        throw cms::Exception("TestFailure") << "thinnedRefFrom(" << refName << ").id() " << refToThinned.id()
                                            << " differs from expectation " << thinnedCollectionID;
      }
      if (not refToThinned.isAvailable()) {
        throw cms::Exception("TestFailure") << "thinnedRefFrom(" << refName << ") is not available when it should be";
      }
      // Check twice to test some possible caching problems.
      // need to account for slimming because going through an explicit ref-to-slimmed
      const int expected = (expectedValue + eventOffset) * thinnedSlimmedValueFactor_;
      if (refToThinned->a != expected) {
        throw cms::Exception("TestFailure") << "Unexpected values from thinnedRefFrom(" << refName << "), got "
                                            << refToThinned->a << " expected " << expected;
      }
      if (refToThinned->a != expected) {
        throw cms::Exception("TestFailure") << "Unexpected values from thinnedRefFrom(" << refName << ") (2nd try) "
                                            << refToThinned->a << " expected " << expected;
      }
    }
  }

  void ThinningTestAnalyzer::testVectors(TrackOfThings const& track,
                                         edm::RefProd<ThingCollection> const& thinnedRefProd,
                                         edm::EDProductGetter const& productGetter,
                                         int const eventOffset) const {
    // Test RefVector, PtrVector, and RefToBaseVector
    unsigned int k = 0;
    bool allPresent = true;
    for (auto iExpectedValue : expectedValues_) {
      if (iExpectedValue != -1) {
        if (not thinnedWasDropped_) {
          auto refToThinned = edm::thinnedRefFrom(track.refVector1[k], thinnedRefProd, productGetter);
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
          if (track.ptrVector1[k]->a != expected) {
            throw cms::Exception("TestFailure") << "unexpected values from ptrVector1";
          }
          if (track.refToBaseVector1[k]->a != expected) {
            throw cms::Exception("TestFailure") << "unexpected values from refToBaseVector1";
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
      if (!track.ptrVector1.isAvailable()) {
        throw cms::Exception("TestFailure") << "unexpected value (false) from ptrVector::isAvailable";
      }
      if (!track.refToBaseVector1.isAvailable()) {
        throw cms::Exception("TestFailure") << "unexpected value (false) from refToBaseVector::isAvailable";
      }
    } else {
      if (track.refVector1.isAvailable()) {
        throw cms::Exception("TestFailure") << "unexpected value (true) from refVector::isAvailable";
      }
      if (track.ptrVector1.isAvailable()) {
        throw cms::Exception("TestFailure") << "unexpected value (true) from ptrVector::isAvailable";
      }
      if (track.refToBaseVector1.isAvailable()) {
        throw cms::Exception("TestFailure") << "unexpected value (true) from refToBaseVector::isAvailable";
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
        if (track.ptrVector1[k]->a != expected) {
          throw cms::Exception("TestFailure") << "unexpected values from ptrVector1";
        }
        if (track.refToBaseVector1[k]->a != expected) {
          throw cms::Exception("TestFailure") << "unexpected values from refToBaseVector1";
        }
      } else {
        allPresent = false;
      }
      ++k;
    }
  }

  void ThinningTestAnalyzer::incrementExpectedValue(std::vector<int>::const_iterator& iter) const {
    ++iter;
    if (iter == expectedValues_.end())
      iter = expectedValues_.begin();
  }
}  // namespace edmtest
using edmtest::ThinningTestAnalyzer;
DEFINE_FWK_MODULE(ThinningTestAnalyzer);
