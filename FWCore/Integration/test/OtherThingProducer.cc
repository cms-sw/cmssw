#include "FWCore/Integration/test/OtherThingProducer.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edmtest {
  OtherThingProducer::OtherThingProducer(edm::ParameterSet const& pset): alg_(), thingLabel_() {
    produces<OtherThingCollection>("testUserTag");
    produces<OtherThingCollection, edm::InLumi>("beginLumi");
    produces<OtherThingCollection, edm::InLumi>("endLumi");
    produces<OtherThingCollection, edm::InRun>("beginRun");
    produces<OtherThingCollection, edm::InRun>("endRun");
    thingLabel_ = pset.getUntrackedParameter<std::string>("thingLabel", std::string("Thing"));
  }

  // Virtual destructor needed.
  OtherThingProducer::~OtherThingProducer() {}  

  // Functions that gets called by framework every event
  void OtherThingProducer::produce(edm::Event& e, edm::EventSetup const&) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<OtherThingCollection> result(new OtherThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(e.me(), *result, thingLabel_);

    // Step D: Put outputs into event
    e.put(result, std::string("testUserTag"));
  }
  // Functions that gets called by framework every luminosity block
  void OtherThingProducer::beginLuminosityBlock(edm::LuminosityBlock& lb, edm::EventSetup const&) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<OtherThingCollection> result(new OtherThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(lb.me(), *result, thingLabel_, std::string("beginLumi"));

    // Step D: Put outputs into lumi block
    lb.put(result, "beginLumi");
  }

  void OtherThingProducer::endLuminosityBlock(edm::LuminosityBlock& lb, edm::EventSetup const&) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<OtherThingCollection> result(new OtherThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(lb.me(), *result, thingLabel_, std::string("endLumi"));

    // Step D: Put outputs into lumi block
    lb.put(result, "endLumi");
  }

  // Functions that gets called by framework every run
  void OtherThingProducer::beginRun(edm::Run& r, edm::EventSetup const&) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<OtherThingCollection> result(new OtherThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(r.me(), *result, thingLabel_, std::string("beginRun"));

    // Step D: Put outputs into event
    r.put(result, "beginRun");
  }

  void OtherThingProducer::endRun(edm::Run& r, edm::EventSetup const&) {
    // Step A: Get Inputs 

    // Step B: Create empty output 
    std::auto_ptr<OtherThingCollection> result(new OtherThingCollection);  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(r.me(), *result, thingLabel_, std::string("endRun"));

    // Step D: Put outputs into event
    r.put(result, "endRun");
  }

}
using edmtest::OtherThingProducer;
DEFINE_FWK_MODULE(OtherThingProducer);
