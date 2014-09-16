#include "FWCore/Integration/test/ThingExtSource.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"

namespace edmtest {
  ThingExtSource::ThingExtSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc) :
    ProducerSourceFromFiles(pset, desc, true), alg_() {
    produces<ThingCollection>();
    produces<ThingCollection, edm::InLumi>("beginLumi");
    produces<ThingCollection, edm::InLumi>("endLumi");
    produces<ThingCollection, edm::InRun>("beginRun");
    produces<ThingCollection, edm::InRun>("endRun");
  }

  // Virtual destructor needed.
  ThingExtSource::~ThingExtSource() { }  

  // Functions that gets called by framework every event
  bool ThingExtSource::setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&) {
    // Fake running out of data.
    if (event() > 2) return false;
    return true;
  }

  void ThingExtSource::produce(edm::Event& e) {

    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    e.put(result);
  }

  // Functions that gets called by framework every luminosity block
  void ThingExtSource::beginLuminosityBlock(edm::LuminosityBlock& lb) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into lumi block
    lb.put(result, "beginLumi");
  }

  void ThingExtSource::endLuminosityBlock(edm::LuminosityBlock& lb) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into lumi block
    lb.put(result, "endLumi");
  }

  // Functions that gets called by framework every run
  void ThingExtSource::beginRun(edm::Run& r) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    r.put(result, "beginRun");
  }

  void ThingExtSource::endRun(edm::Run& r) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    r.put(result, "endRun");
  }

}
using edmtest::ThingExtSource;
DEFINE_FWK_INPUT_SOURCE(ThingExtSource);
