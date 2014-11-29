
#include "FWCore/Framework/interface/EDAnalyzer.h"

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

  class ThinningTestAnalyzer : public edm::EDAnalyzer {
  public:

    explicit ThinningTestAnalyzer(edm::ParameterSet const& pset);

    virtual ~ThinningTestAnalyzer() {}

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    virtual void analyze(edm::Event const& e, edm::EventSetup const& c) override;

  private:

    void incrementExpectedValue(std::vector<int>::const_iterator& iter) const;

    edm::EDGetTokenT<ThingCollection> parentToken_;
    edm::EDGetTokenT<ThingCollection> thinnedToken_;
    edm::EDGetTokenT<edm::ThinnedAssociation> associationToken_;
    edm::EDGetTokenT<TrackOfThingsCollection> trackToken_;

    bool parentWasDropped_;
    std::vector<int> expectedParentContent_;
    bool thinnedWasDropped_;
    bool thinnedIsAlias_;
    std::vector<int> expectedThinnedContent_;
    std::vector<unsigned int> expectedIndexesIntoParent_;
    bool associationShouldBeDropped_;
    std::vector<int> expectedValues_;
  };

  ThinningTestAnalyzer::ThinningTestAnalyzer(edm::ParameterSet const& pset) {
    parentToken_ = consumes<ThingCollection>(pset.getParameter<edm::InputTag>("parentTag"));
    thinnedToken_ = mayConsume<ThingCollection>(pset.getParameter<edm::InputTag>("thinnedTag"));
    associationToken_ = mayConsume<edm::ThinnedAssociation>(pset.getParameter<edm::InputTag>("associationTag"));
    trackToken_ = consumes<TrackOfThingsCollection>(pset.getParameter<edm::InputTag>("trackTag"));
    parentWasDropped_ = pset.getParameter<bool>("parentWasDropped");
    if(!parentWasDropped_) {
      expectedParentContent_ = pset.getParameter<std::vector<int> >("expectedParentContent");
    }
    thinnedWasDropped_ = pset.getParameter<bool>("thinnedWasDropped");
    thinnedIsAlias_ = pset.getParameter<bool>("thinnedIsAlias");
    if(!thinnedWasDropped_) {
      expectedThinnedContent_ = pset.getParameter<std::vector<int> >("expectedThinnedContent");
    }
    associationShouldBeDropped_ = pset.getParameter<bool>("associationShouldBeDropped");
    if(!associationShouldBeDropped_) {
      expectedIndexesIntoParent_ = pset.getParameter<std::vector<unsigned int> >("expectedIndexesIntoParent");
    }
    expectedValues_ = pset.getParameter<std::vector<int> >("expectedValues");
  }

  void ThinningTestAnalyzer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("parentTag");
    desc.add<edm::InputTag>("thinnedTag");
    desc.add<edm::InputTag>("associationTag");
    desc.add<edm::InputTag>("trackTag");
    desc.add<bool>("parentWasDropped", false);
    desc.add<bool>("thinnedWasDropped", false);
    desc.add<bool>("thinnedIsAlias", false);
    std::vector<int> defaultV;
    std::vector<unsigned int> defaultVU;
    desc.add<std::vector<int> >("expectedParentContent", defaultV);
    desc.add<std::vector<int> >("expectedThinnedContent", defaultV);
    desc.add<std::vector<unsigned int> >("expectedIndexesIntoParent", defaultVU);
    desc.add<bool>("associationShouldBeDropped", false);
    desc.add<std::vector<int> >("expectedValues");
    descriptions.addDefault(desc);
  }

  void ThinningTestAnalyzer::analyze(edm::Event const& event, edm::EventSetup const&) {

    edm::Handle<ThingCollection> parentCollection;
    event.getByToken(parentToken_, parentCollection);

    edm::Handle<ThingCollection> thinnedCollection;
    event.getByToken(thinnedToken_, thinnedCollection);

    edm::Handle<edm::ThinnedAssociation> associationCollection;
    event.getByToken(associationToken_, associationCollection);

    edm::Handle<TrackOfThingsCollection> trackCollection;
    event.getByToken(trackToken_, trackCollection);

    unsigned int i = 0;
    if(parentWasDropped_) {
      if(parentCollection.isValid()) {
        throw cms::Exception("TestFailure") << "parent collection present, should have been dropped";
      }
    } else if(!expectedParentContent_.empty()) {
      if(parentCollection->size() != expectedParentContent_.size()) {
        throw cms::Exception("TestFailure") << "parent collection has unexpected size";
      }
      for(auto const& thing : *parentCollection) {
        // Just some numbers that match the somewhat arbitrary values put in
        // by the ThingProducer.
        int expected = static_cast<int>(expectedParentContent_.at(i) + event.eventAuxiliary().event() * 100 + 100);
        if(thing.a != expected) {
          throw cms::Exception("TestFailure") << "parent collection has unexpected content";
        }
        ++i;
      }
    }

    // Check to see the content is what we expect based on what was written
    // by ThingProducer and TrackOfThingsProducer. The values are somewhat
    // arbitrary and meaningless.
    if(thinnedWasDropped_) {
      if(thinnedCollection.isValid()) {
        throw cms::Exception("TestFailure") << "thinned collection present, should have been dropped";
      }
    } else {
      unsigned expectedIndex = 0;
      if(thinnedCollection->size() != expectedThinnedContent_.size()) {
        throw cms::Exception("TestFailure") << "thinned collection has unexpected size";
      }
      for(auto const& thing : *thinnedCollection) {
        if(thing.a != static_cast<int>(expectedThinnedContent_.at(expectedIndex) + event.eventAuxiliary().event() * 100 + 100)) {
          throw cms::Exception("TestFailure") << "thinned collection has unexpected content";
        }
        ++expectedIndex;
      }
    }

    if(associationShouldBeDropped_ && associationCollection.isValid()) {
      throw cms::Exception("TestFailure") << "association collection should have been dropped but was not";
    }
    if(!associationShouldBeDropped_) {
      unsigned int expectedIndex = 0;
      if(associationCollection->indexesIntoParent().size() != expectedIndexesIntoParent_.size()) {
        throw cms::Exception("TestFailure") << "association collection has unexpected size";
      }
      for(auto const& association : associationCollection->indexesIntoParent()) {
        if(association != expectedIndexesIntoParent_.at(expectedIndex)) {
          throw cms::Exception("TestFailure") << "association collection has unexpected content";
        }
        ++expectedIndex;
      }
    }

    if(!parentWasDropped_ && !associationShouldBeDropped_) {
      if(associationCollection->parentCollectionID() != parentCollection.id()) {
        throw cms::Exception("TestFailure") << "analyze parent ProductID is not correct";
      }
    }

    if(!thinnedWasDropped_ && !associationShouldBeDropped_) {
      if(!thinnedIsAlias_) {
        if(associationCollection->thinnedCollectionID() != thinnedCollection.id()) {
          throw cms::Exception("TestFailure") << "analyze thinned ProductID is not correct";
        }
      } else {
        if(associationCollection->thinnedCollectionID() == thinnedCollection.id()) {
          throw cms::Exception("TestFailure") << "analyze thinned ProductID is not correct";
        }
      }
    }

    if(trackCollection->size() != 5u) {
      throw cms::Exception("TestFailure") << "unexpected Track size";
    }

    int eventOffset = static_cast<int>(event.eventAuxiliary().event() * 100 + 100);

    std::vector<int>::const_iterator expectedValue = expectedValues_.begin();
    for(auto const& track : *trackCollection) {

      if(*expectedValue == -1) {
        if(track.ref1.isAvailable()) {
          throw cms::Exception("TestFailure") << "ref1 is available when it should not be";
        }
      } else {
        if(!track.ref1.isAvailable()) {
          throw cms::Exception("TestFailure") << "ref1 is not available when it should be";
        }
        // Check twice to test some possible caching problems.
        if(track.ref1->a !=
           *expectedValue + eventOffset) {
          throw cms::Exception("TestFailure") << "Unexpected values from ref1";
        }
        if(track.ref1->a !=
           *expectedValue + eventOffset) {
          throw cms::Exception("TestFailure") << "Unexpected values from ref1 (2nd try)";
        }
      }

      if(*expectedValue == -1) {
        if(track.refToBase1.isAvailable()) {
          throw cms::Exception("TestFailure") << "refToBase1 is available when it should not be";
        }
      } else {
        if(!track.refToBase1.isAvailable()) {
          throw cms::Exception("TestFailure") << "refToBase1 is not available when it should be";
        }

        // Check twice to test some possible caching problems.
        if(track.refToBase1->a !=
           *expectedValue + eventOffset) {
          throw cms::Exception("TestFailure") << "unexpected values from refToBase1";
        }
        if(track.refToBase1->a !=
           *expectedValue + eventOffset) {
          throw cms::Exception("TestFailure") << "unexpected values from refToBase1";
        }
      }
      incrementExpectedValue(expectedValue);

      if(*expectedValue == -1) {
        if(track.ref2.isAvailable()) {
          throw cms::Exception("TestFailure") << "ref2 is available when it should not be";
        }
      } else {
        if(!track.ref2.isAvailable()) {
          throw cms::Exception("TestFailure") << "ref2 is not available when it should be";
        }

        if(track.ref2->a !=
           *expectedValue + eventOffset) {
          throw cms::Exception("TestFailure") << "unexpected values from ref2";
        }
        if(track.ref2->a !=
           *expectedValue + eventOffset) {
          throw cms::Exception("TestFailure") <<  "unexpected values from ref2";
        }
      }
      incrementExpectedValue(expectedValue);

      if(*expectedValue == -1) {
        if(track.ptr1.isAvailable()) {
          throw cms::Exception("TestFailure") << "ptr1 is available when it should not be";
        }
      } else {
        if(!track.ptr1.isAvailable()) {
          throw cms::Exception("TestFailure") << "ptr1 is not available when it should be";
        }

        if(track.ptr1->a !=
           *expectedValue + eventOffset) {
          throw cms::Exception("TestFailure") << "unexpected values from ptr1";
        }
        if(track.ptr1->a !=
          *expectedValue + eventOffset) {
          throw cms::Exception("TestFailure") << "unexpected values from ptr1 (2)";
        }
      }
      incrementExpectedValue(expectedValue);

      if(*expectedValue == -1) {
        if(track.ptr2.isAvailable()) {
          throw cms::Exception("TestFailure") << "ptr2 is available when it should not be";
        }
      } else {
        if(!track.ptr2.isAvailable()) {
          throw cms::Exception("TestFailure") << "ptr2 is not available when it should be";
        }

        if(track.ptr2->a !=
           *expectedValue + eventOffset) {
          throw cms::Exception("TestFailure") << "unexpected values from ptr2";
        }
        if(track.ptr2->a !=
           *expectedValue + eventOffset) {
          throw cms::Exception("TestFailure") << " unexpected values from ptr2";
        }
      }
      incrementExpectedValue(expectedValue);

      // Test RefVector, PtrVector, and RefToBaseVector
      unsigned int k = 0;
      bool allPresent = true;
      for(auto iExpectedValue : expectedValues_) {

        if(iExpectedValue != -1) {
          if(track.refVector1[k]->a != iExpectedValue + eventOffset) {
            throw cms::Exception("TestFailure") << "unexpected values from refVector1";
          }
          if(track.ptrVector1[k]->a != iExpectedValue + eventOffset) {
            throw cms::Exception("TestFailure") << "unexpected values from ptrVector1";
          }
          if(track.refToBaseVector1[k]->a != iExpectedValue + eventOffset) {
            throw cms::Exception("TestFailure") << "unexpected values from refToBaseVector1";
          }
        } else {
          allPresent = false;
        }
        ++k;
      }

      if(allPresent) {
        if(!track.refVector1.isAvailable()) {
          throw cms::Exception("TestFailure") << "unexpected value (false) from refVector::isAvailable";
        }
        if(!track.ptrVector1.isAvailable()) {
          throw cms::Exception("TestFailure") << "unexpected value (false) from ptrVector::isAvailable";
        }
        if(!track.refToBaseVector1.isAvailable()) {
          throw cms::Exception("TestFailure") << "unexpected value (false) from refToBaseVector::isAvailable";
        }
      } else {
        if(track.refVector1.isAvailable()) {
          throw cms::Exception("TestFailure") << "unexpected value (true) from refVector::isAvailable";
        }
        if(track.ptrVector1.isAvailable()) {
          throw cms::Exception("TestFailure") << "unexpected value (true) from ptrVector::isAvailable";
        }
        if(track.refToBaseVector1.isAvailable()) {
          throw cms::Exception("TestFailure") << "unexpected value (true) from refToBaseVector::isAvailable";
        }
      }
      k = 0;
      for(auto iExpectedValue : expectedValues_) {
        if(iExpectedValue != -1) {
          if(track.refVector1[k]->a != iExpectedValue + eventOffset) {
            throw cms::Exception("TestFailure") << "unexpected values from refVector1";
          }
          if(track.ptrVector1[k]->a != iExpectedValue + eventOffset) {
            throw cms::Exception("TestFailure") << "unexpected values from ptrVector1";
          }
          if(track.refToBaseVector1[k]->a != iExpectedValue + eventOffset) {
            throw cms::Exception("TestFailure") << "unexpected values from refToBaseVector1";
          }
        } else {
          allPresent = false;
        }
        ++k;
      }
    }
  }

  void ThinningTestAnalyzer::incrementExpectedValue(std::vector<int>::const_iterator& iter) const {
    ++iter;
    if(iter == expectedValues_.end()) iter = expectedValues_.begin();
  }
}
using edmtest::ThinningTestAnalyzer;
DEFINE_FWK_MODULE(ThinningTestAnalyzer);
