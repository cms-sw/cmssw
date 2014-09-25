/** \class edm::ThinningProducer
\author W. David Dagenhart, created 11 June 2014
*/

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/TrackOfThings.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edm {
  class EventSetup;
}

namespace edmtest {

  class TrackOfThingsProducer : public edm::one::EDProducer<> {
  public:

    explicit TrackOfThingsProducer(edm::ParameterSet const&);
    virtual ~TrackOfThingsProducer();

    void produce(edm::Event&, edm::EventSetup const&) override;

  private:

    void incrementKey(std::vector<unsigned int>::const_iterator& key) const;

    edm::EDGetTokenT<ThingCollection> inputToken_;
    std::vector<unsigned int> keysToReference_;
  };

  TrackOfThingsProducer::TrackOfThingsProducer(edm::ParameterSet const& pset) {

    inputToken_ = consumes<ThingCollection>(pset.getParameter<edm::InputTag>("inputTag"));

    keysToReference_ = pset.getParameter<std::vector<unsigned int> >("keysToReference");

    produces<TrackOfThingsCollection>();
  }

  TrackOfThingsProducer::~TrackOfThingsProducer() { }

  void TrackOfThingsProducer::incrementKey(std::vector<unsigned int>::const_iterator& key) const {
    ++key;
    if(key == keysToReference_.end()) key = keysToReference_.begin();
  }

  void TrackOfThingsProducer::produce(edm::Event& event, edm::EventSetup const&) {

    edm::Handle<ThingCollection> inputCollection;
    event.getByToken(inputToken_, inputCollection);

    std::auto_ptr<TrackOfThingsCollection> result(new TrackOfThingsCollection);

    // Arbitrarily fabricate some fake data with TrackOfThings pointing to
    // Thing objects in products written to the event by a different module.
    // The numbers in the keys here are made up, passed in via the configuration
    // and have no meaning other than that we will later check that we can
    // read out what we put in here.
    std::vector<unsigned int>::const_iterator key = keysToReference_.begin();
    std::vector<unsigned int>::const_iterator keyEnd = keysToReference_.end();
    for(unsigned int i = 0; i < 5; ++i) {


      edmtest::TrackOfThings trackOfThings;

      trackOfThings.ref1 = edm::Ref<ThingCollection>(inputCollection, *key);
      incrementKey(key);

      trackOfThings.ref2 = edm::Ref<ThingCollection>(inputCollection, *key);
      incrementKey(key);

      trackOfThings.ptr1 = edm::Ptr<Thing>(inputCollection, *key);
      incrementKey(key);

      trackOfThings.ptr2 = edm::Ptr<Thing>(inputCollection, *key);
      incrementKey(key);

      trackOfThings.refToBase1 = edm::RefToBase<Thing>(trackOfThings.ref1);

      for(auto iKey : keysToReference_) {
        trackOfThings.refVector1.push_back(edm::Ref<ThingCollection>(inputCollection, iKey));
        trackOfThings.ptrVector1.push_back(edm::Ptr<Thing>(inputCollection, iKey));
        trackOfThings.refToBaseVector1.push_back(edm::RefToBase<Thing>(edm::Ref<ThingCollection>(inputCollection, iKey)));
      }

      result->push_back(trackOfThings);
    }

    event.put(result);
  }
}
using edmtest::TrackOfThingsProducer;
DEFINE_FWK_MODULE(TrackOfThingsProducer);
