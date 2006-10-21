#include "FWCore/Integration/test/ThingSource.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"

namespace edmtest {
  ThingSource::ThingSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc) :
    GeneratedInputSource(pset, desc), alg_() {
    produces<ThingCollection>();
  }

  // Virtual destructor needed.
  ThingSource::~ThingSource() { }  

  // Functions that gets called by framework every event
  bool ThingSource::produce(edm::Event& e) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    e.put(result);

    return true;
  }
}
using edmtest::ThingSource;
DEFINE_FWK_INPUT_SOURCE(ThingSource);
