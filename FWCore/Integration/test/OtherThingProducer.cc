#include "FWCore/Integration/test/OtherThingProducer.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {
  OtherThingProducer::OtherThingProducer(edm::ParameterSet const& pset): alg_(), refsAreTransient_(false) {
    produces<OtherThingCollection>("testUserTag");
    useRefs_ = pset.getUntrackedParameter<bool>("useRefs");
    if(useRefs_) {
      thingToken_=consumes<ThingCollection>(pset.getParameter<edm::InputTag>("thingTag"));
    }
    refsAreTransient_ = pset.getUntrackedParameter<bool>("transient");
  }

  // Virtual destructor needed.
  OtherThingProducer::~OtherThingProducer() {}  

  // Functions that gets called by framework every event
  void OtherThingProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<OtherThingCollection> result(new OtherThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(e, *result, thingToken_, useRefs_, refsAreTransient_);

    // Step D: Put outputs into event
    e.put(result, std::string("testUserTag"));
  }
  
  void OtherThingProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("thingTag",edm::InputTag("Thing"))->setComment("Where to get the ThingCollection");
    desc.addUntracked<bool>("useRefs",true)->setComment("Actually get the ThingCollection and build edm::Refs to the contained items.");
    desc.addUntracked<bool>("transient",false)->setComment("If true, then the Refs constructed by the ThingCollection can not be persisted");
    descriptions.add("otherThingProd", desc);
  }

}
using edmtest::OtherThingProducer;
DEFINE_FWK_MODULE(OtherThingProducer);
