#include <string>

#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "OtherThingAlgorithm.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {
  class OtherThingProducer : public edm::global::EDProducer<> {
  public:
    explicit OtherThingProducer(edm::ParameterSet const& ps);

    ~OtherThingProducer() override;

    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    OtherThingAlgorithm alg_;
    edm::EDGetToken thingToken_;
    edm::EDPutToken putToken_;
    bool useRefs_;
    bool refsAreTransient_;
  };

  OtherThingProducer::OtherThingProducer(edm::ParameterSet const& pset) : alg_(), refsAreTransient_(false) {
    putToken_ = produces<OtherThingCollection>("testUserTag");
    useRefs_ = pset.getUntrackedParameter<bool>("useRefs");
    if (useRefs_) {
      thingToken_ = consumes<ThingCollection>(pset.getParameter<edm::InputTag>("thingTag"));
    }
    refsAreTransient_ = pset.getUntrackedParameter<bool>("transient");
  }

  // Virtual destructor needed.
  OtherThingProducer::~OtherThingProducer() {}

  // Functions that gets called by framework every event
  void OtherThingProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // Step A: Get Inputs

    // Step B: Create empty output
    auto result = std::make_unique<OtherThingCollection>();  //Empty

    // Step C: Get data for algorithm
    edm::Handle<ThingCollection> parentHandle;
    if (useRefs_) {
      bool succeeded = e.getByToken(thingToken_, parentHandle);
      assert(succeeded);
      assert(parentHandle.isValid());
    }

    // Step D: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(parentHandle, *result, useRefs_, refsAreTransient_);

    // Step E: Put outputs into event
    e.put(putToken_, std::move(result));
  }

  void OtherThingProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("thingTag", edm::InputTag("Thing"))->setComment("Where to get the ThingCollection");
    desc.addUntracked<bool>("useRefs", true)
        ->setComment("Actually get the ThingCollection and build edm::Refs to the contained items.");
    desc.addUntracked<bool>("transient", false)
        ->setComment("If true, then the Refs constructed by the ThingCollection can not be persisted");
    descriptions.add("otherThingProd", desc);
  }

}  // namespace edmtest
using edmtest::OtherThingProducer;
DEFINE_FWK_MODULE(OtherThingProducer);
