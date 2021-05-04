#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/TrackOfDSVThings.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edm {
  class EventSetup;
}

namespace edmtest {

  class TrackOfDSVThingsProducer : public edm::global::EDProducer<> {
  public:
    explicit TrackOfDSVThingsProducer(edm::ParameterSet const&);

    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  private:
    void incrementKey(std::vector<unsigned int>::const_iterator& key) const;

    const edm::EDGetTokenT<edmNew::DetSetVector<Thing>> inputToken_;
    const edm::EDPutTokenT<TrackOfDSVThingsCollection> outputToken_;
    const std::vector<unsigned int> keysToReference_;
    const unsigned int nTracks_;
  };

  TrackOfDSVThingsProducer::TrackOfDSVThingsProducer(edm::ParameterSet const& pset)
      : inputToken_(consumes<edmNew::DetSetVector<Thing>>(pset.getParameter<edm::InputTag>("inputTag"))),
        outputToken_(produces<TrackOfDSVThingsCollection>()),
        keysToReference_(pset.getParameter<std::vector<unsigned int>>("keysToReference")),
        nTracks_(pset.getParameter<unsigned int>("nTracks")) {}

  void TrackOfDSVThingsProducer::incrementKey(std::vector<unsigned int>::const_iterator& key) const {
    ++key;
    if (key == keysToReference_.end())
      key = keysToReference_.begin();
  }

  void TrackOfDSVThingsProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
    edm::Handle<edmNew::DetSetVector<Thing>> inputCollection = event.getHandle(inputToken_);

    TrackOfDSVThingsCollection result;

    // Arbitrarily fabricate some fake data with TrackOfThings pointing to
    // Thing objects in products written to the event by a different module.
    // The numbers in the keys here are made up, passed in via the configuration
    // and have no meaning other than that we will later check that we can
    // read out what we put in here.
    std::vector<unsigned int>::const_iterator key = keysToReference_.begin();
    for (unsigned int i = 0; i < nTracks_; ++i) {
      edmtest::TrackOfDSVThings trackOfThings;

      trackOfThings.ref1 = edm::Ref<edmNew::DetSetVector<Thing>, Thing>(inputCollection, *key);
      incrementKey(key);

      trackOfThings.ref2 = edm::Ref<edmNew::DetSetVector<Thing>, Thing>(inputCollection, *key);
      incrementKey(key);

      for (auto iKey : keysToReference_) {
        trackOfThings.refVector1.push_back(edm::Ref<edmNew::DetSetVector<Thing>, Thing>(inputCollection, iKey));
      }

      result.push_back(trackOfThings);
    }

    event.emplace(outputToken_, std::move(result));
  }
}  // namespace edmtest
using edmtest::TrackOfDSVThingsProducer;
DEFINE_FWK_MODULE(TrackOfDSVThingsProducer);
