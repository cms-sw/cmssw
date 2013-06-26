#include "FWCore/Integration/test/ThingSource.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"

namespace edmtest {
  ThingSource::ThingSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc) :
    ProducerSourceBase(pset, desc, false), alg_() {
    produces<ThingCollection>();
    produces<ThingCollection, edm::InLumi>("beginLumi");
    produces<ThingCollection, edm::InLumi>("endLumi");
    produces<ThingCollection, edm::InRun>("beginRun");
    produces<ThingCollection, edm::InRun>("endRun");
  }

  // Virtual destructor needed.
  ThingSource::~ThingSource() { }  

  // Functions that gets called by framework every event
  void ThingSource::produce(edm::Event& e) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    e.put(result);
  }

  // Functions that gets called by framework every luminosity block
  void ThingSource::beginLuminosityBlock(edm::LuminosityBlock& lb) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into lumi block
    lb.put(result, "beginLumi");
  }

  void ThingSource::endLuminosityBlock(edm::LuminosityBlock& lb) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into lumi block
    lb.put(result, "endLumi");
  }

  // Functions that gets called by framework every run
  void ThingSource::beginRun(edm::Run& r) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    r.put(result, "beginRun");
  }

  void ThingSource::endRun(edm::Run& r) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    r.put(result, "endRun");
  }
}
using edmtest::ThingSource;
DEFINE_FWK_INPUT_SOURCE(ThingSource);
