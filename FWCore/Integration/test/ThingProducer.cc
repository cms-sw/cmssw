#include "FWCore/FWCoreIntegration/test/ThingProducer.h"
#include "FWCore/FWCoreIntegration/test/ThingCollection.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"

namespace edmreftest {
  ThingProducer::ThingProducer(edm::ParameterSet const&): alg_() {}

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
DEFINE_FWK_MODULE(ThingProducer)
}
