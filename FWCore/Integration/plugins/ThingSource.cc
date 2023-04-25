#include "ThingSource.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"

namespace edmtest {
  ThingSource::ThingSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc)
      : ProducerSourceBase(pset, desc, false), alg_() {
    produces<ThingCollection>();
    produces<ThingCollection, edm::Transition::BeginLuminosityBlock>("beginLumi");
    produces<ThingCollection, edm::Transition::BeginLuminosityBlock>("endLumi");
    produces<ThingCollection, edm::Transition::BeginRun>("beginRun");
    produces<ThingCollection, edm::Transition::BeginRun>("endRun");
  }

  // Virtual destructor needed.
  ThingSource::~ThingSource() {}

  // Functions that gets called by framework every event
  void ThingSource::produce(edm::Event& e) {
    // Step A: Get Inputs

    // Step B: Create empty output
    auto result = std::make_unique<ThingCollection>();  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    e.put(std::move(result));
  }

  // Functions that gets called by framework every luminosity block
  void ThingSource::beginLuminosityBlock(edm::LuminosityBlock& lb) {
    // Step A: Get Inputs

    // Step B: Create empty output
    auto result = std::make_unique<ThingCollection>();  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into lumi block
    lb.put(std::move(result), "beginLumi");

    endLuminosityBlock(lb);
  }

  void ThingSource::endLuminosityBlock(edm::LuminosityBlock& lb) {
    // Step A: Get Inputs

    // Step B: Create empty output
    auto result = std::make_unique<ThingCollection>();  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into lumi block
    lb.put(std::move(result), "endLumi");
  }

  // Functions that gets called by framework every run
  void ThingSource::beginRun(edm::Run& r) {
    // Step A: Get Inputs

    // Step B: Create empty output
    auto result = std::make_unique<ThingCollection>();  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    r.put(std::move(result), "beginRun");

    endRun(r);
  }

  void ThingSource::endRun(edm::Run& r) {
    // Step A: Get Inputs

    // Step B: Create empty output
    auto result = std::make_unique<ThingCollection>();  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    r.put(std::move(result), "endRun");
  }

  void ThingSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("Creates ThingCollections for testing.");
    edm::ProducerSourceBase::fillDescription(desc);
    descriptions.add("source", desc);
  }

}  // namespace edmtest
using edmtest::ThingSource;
DEFINE_FWK_INPUT_SOURCE(ThingSource);
