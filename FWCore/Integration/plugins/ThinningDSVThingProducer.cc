#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ThinnedRefSet.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/TrackOfDSVThings.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/ThinningProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <string>

namespace edmtest {

  class ThinningDSVThingSelector {
  public:
    ThinningDSVThingSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc);

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

    void preChoose(edm::Handle<edmNew::DetSetVector<Thing>> tc, edm::Event const& event, edm::EventSetup const& es);

    bool choose(unsigned int iIndex, edmtest::Thing const& iItem) const;

    void reset() { keysToSave_.clear(); }

  private:
    edm::EDGetTokenT<TrackOfDSVThingsCollection> const trackToken_;
    edm::ThinnedRefSet<edmNew::DetSetVector<Thing>> keysToSave_;
    unsigned int const offsetToValue_;
    unsigned int const expectedDetSets_;
    unsigned int const expectedDetSetSize_;
    int const slimmedValueFactor_;
  };

  ThinningDSVThingSelector::ThinningDSVThingSelector(edm::ParameterSet const& pset, edm::ConsumesCollector&& cc)
      : trackToken_(cc.consumes<TrackOfDSVThingsCollection>(pset.getParameter<edm::InputTag>("trackTag"))),
        keysToSave_(pset.getParameter<bool>("thinnedRefSetIgnoreInvalidParentRef")
                        ? edm::ThinnedRefSetMode::ignoreInvalidParentRef
                        : edm::ThinnedRefSetMode::throwOnInvalidParentRef),
        offsetToValue_(pset.getParameter<unsigned int>("offsetToValue")),
        expectedDetSets_(pset.getParameter<unsigned int>("expectedDetSets")),
        expectedDetSetSize_(pset.getParameter<unsigned int>("expectedDetSetSize")),
        slimmedValueFactor_(pset.getParameter<int>("slimmedValueFactor")) {}

  void ThinningDSVThingSelector::fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.add<edm::InputTag>("trackTag");
    desc.add<unsigned int>("offsetToValue", 0);
    desc.add<unsigned int>("expectedDetSets");
    desc.add<unsigned int>("expectedDetSetSize");
    desc.add<int>("slimmedValueFactor", 1);
    desc.add<bool>("thinnedRefSetIgnoreInvalidParentRef", false);
  }

  void ThinningDSVThingSelector::preChoose(edm::Handle<edmNew::DetSetVector<Thing>> tc,
                                           edm::Event const& event,
                                           edm::EventSetup const& es) {
    auto filler = keysToSave_.fill(edm::RefProd(tc), event.productGetter());
    for (auto const& track : event.get(trackToken_)) {
      filler.insert(track.ref1);
      filler.insert(track.ref2);
    }

    // Just checking to see if the collection got passed in. Not really using it for anything.
    if (tc->size() != expectedDetSets_) {
      throw cms::Exception("TestFailure") << "ThinningDSVThingSelector::preChoose, number of DetSets = " << tc->size()
                                          << " expected = " << expectedDetSets_;
    }
    for (auto const& ds : *tc) {
      if (ds.size() != expectedDetSetSize_) {
        throw cms::Exception("TestFailure")
            << "ThinningDSVThingSelector::preChoose, number of elements in DetSet with id " << ds.id() << " = "
            << ds.size() << " expected = " << expectedDetSetSize_;
      }
    }
  }

  bool ThinningDSVThingSelector::choose(unsigned int iIndex, edmtest::Thing const& iItem) const {
    // Just checking to see the element in the container got passed in OK. Not really using it.
    // Just using %10 because it coincidentally works with the arbitrary numbers I picked, no meaning really.
    auto const expected = slimmedValueFactor_ * (iIndex + offsetToValue_);
    if (static_cast<unsigned>(iItem.a % 10) != static_cast<unsigned>(expected % 10)) {
      throw cms::Exception("TestFailure") << "ThinningDSVThingSelector::choose, item content = " << iItem.a
                                          << " index = " << iIndex << " expected " << expected;
    }

    // Save the Things referenced by the Tracks
    return keysToSave_.contains(iIndex);
  }
}  // namespace edmtest

using ThinningDSVThingProducer =
    edm::ThinningProducer<edmNew::DetSetVector<edmtest::Thing>, edmtest::ThinningDSVThingSelector>;
DEFINE_FWK_MODULE(ThinningDSVThingProducer);
