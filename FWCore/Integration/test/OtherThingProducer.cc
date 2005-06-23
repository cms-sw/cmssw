#include "FWCore/FWCoreIntegration/test/OtherThingProducer.h"
#include "FWCore/FWCoreIntegration/test/OtherThingCollection.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"


namespace edmreftest {
  OtherThingProducer::OtherThingProducer(edm::ParameterSet const&): alg_() {}

  // Virtual destructor needed.
  OtherThingProducer::~OtherThingProducer() {}  

  // Functions that gets called by framework every event
  void OtherThingProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<OtherThingCollection> result(new OtherThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(e, *result);

    // Step D: Put outputs into event
    e.put(result);
  }
DEFINE_FWK_MODULE(OtherThingProducer)
}
