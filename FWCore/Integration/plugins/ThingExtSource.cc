#include "ThingExtSource.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"

namespace edmtest {
  ThingExtSource::ThingExtSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc)
      : ProducerSourceFromFiles(pset, desc, true), alg_() {
    produces<ThingCollection>();
    produces<ThingCollection, edm::Transition::BeginLuminosityBlock>("beginLumi");
    produces<ThingCollection, edm::Transition::BeginLuminosityBlock>("endLumi");
    produces<ThingCollection, edm::Transition::BeginRun>("beginRun");
    produces<ThingCollection, edm::Transition::BeginRun>("endRun");
  }

  // Virtual destructor needed.
  ThingExtSource::~ThingExtSource() {}

  // Functions that gets called by framework every event
  bool ThingExtSource::setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&, edm::EventAuxiliary::ExperimentType&) {
    // Fake running out of data.
    if (event() > 2)
      return false;
    return true;
  }

  void ThingExtSource::produce(edm::Event& e) {
    // Step A: Get Inputs

    // Step B: Create empty output
    auto result = std::make_unique<ThingCollection>();  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    e.put(std::move(result));
  }

  // Functions that gets called by framework every luminosity block
  void ThingExtSource::beginLuminosityBlock(edm::LuminosityBlock& lb) {
    // Step A: Get Inputs

    // Step B: Create empty output
    auto result = std::make_unique<ThingCollection>();  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into lumi block
    lb.put(std::move(result), "beginLumi");

    endLuminosityBlock(lb);
  }

  void ThingExtSource::endLuminosityBlock(edm::LuminosityBlock& lb) {
    // Step A: Get Inputs

    // Step B: Create empty output
    auto result = std::make_unique<ThingCollection>();  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into lumi block
    lb.put(std::move(result), "endLumi");
  }

  // Functions that gets called by framework every run
  void ThingExtSource::beginRun(edm::Run& r) {
    // Step A: Get Inputs

    // Step B: Create empty output
    auto result = std::make_unique<ThingCollection>();  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    r.put(std::move(result), "beginRun");

    endRun(r);
  }

  void ThingExtSource::endRun(edm::Run& r) {
    // Step A: Get Inputs

    // Step B: Create empty output
    auto result = std::make_unique<ThingCollection>();  //Empty

    // Step C: Invoke the algorithm, passing in inputs (NONE) and getting back outputs.
    alg_.run(*result);

    // Step D: Put outputs into event
    r.put(std::move(result), "endRun");
  }

  void ThingExtSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("Creates ThingCollections from a file for testing.");
    edm::ProducerSourceFromFiles::fillDescription(desc);
    descriptions.add("source", desc);
  }

}  // namespace edmtest
using edmtest::ThingExtSource;
DEFINE_FWK_INPUT_SOURCE(ThingExtSource);
