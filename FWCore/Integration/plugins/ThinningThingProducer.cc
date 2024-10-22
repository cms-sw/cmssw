
/** \class edm::ThinningThingSelector
\author W. David Dagenhart, created 11 June 2014
*/

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

  class ThinningThingSelector {
  public:
    ThinningThingSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc);

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

    void preChoose(edm::Handle<edmtest::ThingCollection> tc, edm::Event const& event, edm::EventSetup const& es);

    bool choose(unsigned int iIndex, edmtest::Thing const& iItem);

    void reset() { keysToSave_.clear(); }

  private:
    edm::EDGetTokenT<TrackOfThingsCollection> trackToken_;
    edm::ESGetToken<edmtest::WhatsIt, GadgetRcd> whatsItToken_;

    std::set<unsigned int> keysToSave_;
    unsigned int offsetToThinnedKey_;
    unsigned int offsetToValue_;
    unsigned int expectedCollectionSize_;
    int slimmedValueFactor_;
  };

  ThinningThingSelector::ThinningThingSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc) {
    trackToken_ = cc.consumes<TrackOfThingsCollection>(pset.getParameter<edm::InputTag>("trackTag"));
    whatsItToken_ = cc.esConsumes();
    offsetToThinnedKey_ = pset.getParameter<unsigned int>("offsetToThinnedKey");
    offsetToValue_ = pset.getParameter<unsigned int>("offsetToValue");
    expectedCollectionSize_ = pset.getParameter<unsigned int>("expectedCollectionSize");
    slimmedValueFactor_ = pset.getParameter<int>("slimmedValueFactor");
  }

  void ThinningThingSelector::fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.add<edm::InputTag>("trackTag");
    desc.add<unsigned int>("offsetToThinnedKey");
    desc.add<unsigned int>("offsetToValue", 0);
    desc.add<unsigned int>("expectedCollectionSize");
    desc.add<int>("slimmedValueFactor", 1);
  }

  void ThinningThingSelector::preChoose(edm::Handle<edmtest::ThingCollection> tc,
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
      throw cms::Exception("TestFailure") << "ThinningThingSelector::preChoose, collection size = " << tc->size()
                                          << " expected size = " << expectedCollectionSize_;
    }

    // Just checking to see the EventSetup works from here. Not really using it for anything.
    edm::ESHandle<edmtest::WhatsIt> pSetup = es.getHandle(whatsItToken_);
    pSetup.isValid();
  }

  bool ThinningThingSelector::choose(unsigned int iIndex, edmtest::Thing const& iItem) {
    // Just checking to see the element in the container got passed in OK. Not really using it.
    // Just using %10 because it coincidentally works with the arbitrary numbers I picked, no meaning really.
    auto const expected = slimmedValueFactor_ * (iIndex + offsetToValue_);
    if (static_cast<unsigned>(iItem.a % 10) != static_cast<unsigned>(expected % 10)) {
      throw cms::Exception("TestFailure") << "ThinningThingSelector::choose, item content = " << iItem.a
                                          << " index = " << iIndex << " expected " << expected;
    }

    // Save the Things referenced by the Tracks
    if (keysToSave_.find(iIndex) == keysToSave_.end())
      return false;
    return true;
  }
}  // namespace edmtest

typedef edm::ThinningProducer<edmtest::ThingCollection, edmtest::ThinningThingSelector> ThinningThingProducer;
DEFINE_FWK_MODULE(ThinningThingProducer);
