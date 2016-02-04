#include "FWCore/Integration/test/OtherThingProducer.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edmtest {
  OtherThingProducer::OtherThingProducer(edm::ParameterSet const& pset): alg_(), thingLabel_(), refsAreTransient_(false) {
    produces<OtherThingCollection>("testUserTag");
    thingLabel_ = pset.getUntrackedParameter<std::string>("thingLabel", std::string("Thing"));
    refsAreTransient_ = pset.getUntrackedParameter<bool>("transient", false);
  }

  // Virtual destructor needed.
  OtherThingProducer::~OtherThingProducer() {}  

  // Functions that gets called by framework every event
  void OtherThingProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<OtherThingCollection> result(new OtherThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(e, *result, thingLabel_, std::string(), refsAreTransient_);

    // Step D: Put outputs into event
    e.put(result, std::string("testUserTag"));
  }
}
using edmtest::OtherThingProducer;
DEFINE_FWK_MODULE(OtherThingProducer);
