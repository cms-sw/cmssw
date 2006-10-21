#include "FWCore/Integration/test/ThingExtSource.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"

namespace edmtest {
  ThingExtSource::ThingExtSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc) :
    ExternalInputSource(pset, desc), alg_() {
    produces<ThingCollection>();
  }

  // Virtual destructor needed.
  ThingExtSource::~ThingExtSource() { }  

  // Functions that gets called by framework every event
  bool ThingExtSource::produce(edm::Event& e) {

    // Fake running out of data for an external input source.
    if (event() > 2) return false;

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
using edmtest::ThingExtSource;
DEFINE_FWK_INPUT_SOURCE(ThingExtSource);
