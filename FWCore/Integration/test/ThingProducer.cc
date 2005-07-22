#include "FWCore/Integration/test/ThingProducer.h"
#include "FWCore/Integration/test/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edmreftest {
  ThingProducer::ThingProducer(edm::ParameterSet const&): alg_() {
    produces<ThingCollection>();
  }

  // Virtual destructor needed.
  ThingProducer::~ThingProducer() { }  

  // Functions that gets called by framework every event
  void ThingProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    e.put(result);
  }
}
using edmreftest::ThingProducer;
DEFINE_FWK_MODULE(ThingProducer)
