#include "FWCore/Integration/test/ThingProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edmtest {
  ThingProducer::ThingProducer(edm::ParameterSet const& iConfig)
      : alg_(iConfig.getParameter<int>("offsetDelta"),
             iConfig.getParameter<int>("nThings"),
             iConfig.getParameter<bool>("grow")),
        noPut_(iConfig.getUntrackedParameter<bool>("noPut"))  // used for testing with missing products
  {
    evToken_ = produces<ThingCollection>();
    blToken_ = produces<ThingCollection, edm::Transition::BeginLuminosityBlock>("beginLumi");
    elToken_ = produces<ThingCollection, edm::Transition::EndLuminosityBlock>("endLumi");
    brToken_ = produces<ThingCollection, edm::Transition::BeginRun>("beginRun");
    erToken_ = produces<ThingCollection, edm::Transition::EndRun>("endRun");
  }

  // Virtual destructor needed.
  ThingProducer::~ThingProducer() {}

  // Functions that gets called by framework every event
  void ThingProducer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
    // Step A: Get Inputs

    // Step B: Create empty output
    ThingCollection result;  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(result);

    // Step D: Put outputs into event
    if (!noPut_)
      e.emplace(evToken_, std::move(result));
  }

  // Functions that gets called by framework every luminosity block
  void ThingProducer::globalBeginLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const&) const {
    // Step A: Get Inputs

    // Step B: Create empty output
    ThingCollection result;  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(result);

    // Step D: Put outputs into lumi block
    if (!noPut_)
      lb.emplace(blToken_, std::move(result));
  }

  void ThingProducer::globalEndLuminosityBlockProduce(edm::LuminosityBlock& lb, edm::EventSetup const&) const {
    // Step A: Get Inputs

    // Step B: Create empty output
    ThingCollection result;  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(result);

    // Step D: Put outputs into lumi block
    if (!noPut_)
      lb.emplace(elToken_, std::move(result));
  }

  // Functions that gets called by framework every run
  void ThingProducer::globalBeginRunProduce(edm::Run& r, edm::EventSetup const&) const {
    // Step A: Get Inputs

    // Step B: Create empty output
    ThingCollection result;  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(result);

    // Step D: Put outputs into event
    if (!noPut_)
      r.emplace(brToken_, std::move(result));
  }

  void ThingProducer::globalEndRunProduce(edm::Run& r, edm::EventSetup const&) const {
    // Step A: Get Inputs

    // Step B: Create empty output
    ThingCollection result;  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(result);

    // Step D: Put outputs into event
    if (!noPut_)
      r.emplace(erToken_, std::move(result));
  }

  void ThingProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<int>("offsetDelta", 0)
        ->setComment(
            "How much extra to increment the value used when creating Things for a new container. E.g. the last value "
            "used to create Thing from the previous event is incremented by 'offsetDelta' to compute the value to use "
            "of the first Thing created in the next Event.");
    desc.add<int>("nThings", 20)->setComment("How many Things to put in each collection.");
    desc.add<bool>("grow", false)
        ->setComment("If true, multiply 'nThings' by the value of offset for each run of the algorithm.");
    desc.addUntracked<bool>("noPut", false)
        ->setComment("If true, data is not put into the Principal. This is used to test missing products.");
    descriptions.add("thingProd", desc);
  }

}  // namespace edmtest
using edmtest::ThingProducer;
DEFINE_FWK_MODULE(ThingProducer);
