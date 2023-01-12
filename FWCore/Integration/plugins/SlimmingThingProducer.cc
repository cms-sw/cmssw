#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/TrackOfThings.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/ThinningProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "WhatsIt.h"
#include "GadgetRcd.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <set>
#include <string>

namespace edmtest {

  class SlimmingThingSelector {
  public:
    SlimmingThingSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc);

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

    void preChoose(edm::Handle<edmtest::ThingCollection> tc, edm::Event const& event, edm::EventSetup const& es);

    std::optional<edmtest::Thing> choose(unsigned int iIndex, edmtest::Thing const& iItem) const;

    void reset() { keysToSave_.clear(); }

  private:
    edm::EDGetTokenT<TrackOfThingsCollection> const trackToken_;
    edm::ESGetToken<edmtest::WhatsIt, GadgetRcd> const setupToken_;
    std::set<unsigned int> keysToSave_;
    unsigned int const offsetToThinnedKey_;
    unsigned int const offsetToValue_;
    unsigned int const expectedCollectionSize_;
    int const slimmedValueFactor_;
  };

  SlimmingThingSelector::SlimmingThingSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc)
      : trackToken_(cc.consumes<TrackOfThingsCollection>(pset.getParameter<edm::InputTag>("trackTag"))),
        setupToken_(cc.esConsumes<edmtest::WhatsIt, GadgetRcd>()),
        offsetToThinnedKey_(pset.getParameter<unsigned int>("offsetToThinnedKey")),
        offsetToValue_(pset.getParameter<unsigned int>("offsetToValue")),
        expectedCollectionSize_(pset.getParameter<unsigned int>("expectedCollectionSize")),
        slimmedValueFactor_(pset.getParameter<int>("slimmedValueFactor")) {}

  void SlimmingThingSelector::fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.add<edm::InputTag>("trackTag");
    desc.add<unsigned int>("offsetToThinnedKey");
    desc.add<unsigned int>("offsetToValue", 0);
    desc.add<unsigned int>("expectedCollectionSize");
    desc.add<int>("slimmedValueFactor", 1);
  }

  void SlimmingThingSelector::preChoose(edm::Handle<edmtest::ThingCollection> tc,
                                        edm::Event const& event,
                                        edm::EventSetup const& es) {
    for (auto const& track : event.get(trackToken_)) {
      keysToSave_.insert(track.ref1.key() - offsetToThinnedKey_);
      keysToSave_.insert(track.ref2.key() - offsetToThinnedKey_);
      keysToSave_.insert(track.ptr1.key() - offsetToThinnedKey_);
      keysToSave_.insert(track.ptr2.key() - offsetToThinnedKey_);
    }

    // Just checking to see if the collection got passed in. Not really using it for anything.
    if (tc->size() != expectedCollectionSize_) {
      throw cms::Exception("TestFailure") << "SlimmingThingSelector::preChoose, collection size = " << tc->size()
                                          << " expected size = " << expectedCollectionSize_;
    }

    // Just checking to see the EventSetup works from here. Not really using it for anything.
    edm::ESHandle<edmtest::WhatsIt> pSetup = es.getHandle(setupToken_);
    pSetup.isValid();
  }

  std::optional<edmtest::Thing> SlimmingThingSelector::choose(unsigned int iIndex, edmtest::Thing const& iItem) const {
    // Just checking to see the element in the container got passed in OK. Not really using it.
    // Just using %10 because it coincidentally works with the arbitrary numbers I picked, no meaning really.
    auto const expected = slimmedValueFactor_ * (iIndex + offsetToValue_);
    if (static_cast<unsigned>(iItem.a % 10) != static_cast<unsigned>(expected % 10)) {
      throw cms::Exception("TestFailure") << "SlimmingThingSelector::choose, item content = " << iItem.a
                                          << " index = " << iIndex << " expected " << expected;
    }

    // Save the Things referenced by the Tracks
    if (keysToSave_.find(iIndex) == keysToSave_.end())
      return {};
    auto copy = iItem;
    copy.a *= 10;
    return copy;
  }
}  // namespace edmtest

typedef edm::ThinningProducer<edmtest::ThingCollection, edmtest::SlimmingThingSelector> SlimmingThingProducer;
DEFINE_FWK_MODULE(SlimmingThingProducer);
