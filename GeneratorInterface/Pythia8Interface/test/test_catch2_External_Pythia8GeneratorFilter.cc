#include "catch.hpp"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

static constexpr auto s_tag = "[External Pythia8GeneratorFilter]";

TEST_CASE("Standard checks of ExternalGeneratorFilter with Pythia8GeneratorFilter", s_tag) {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.RandomNumberGeneratorService.toTest = process.RandomNumberGeneratorService.generator.clone()
from GeneratorInterface.Core.ExternalGeneratorFilter import ExternalGeneratorFilter
_pythia8 = cms.EDFilter("Pythia8GeneratorFilter",
    comEnergy = cms.double(7000.),
    PythiaParameters = cms.PSet(
        pythia8_example02 = cms.vstring('HardQCD:all = on',
                                        'PhaseSpace:pTHatMin = 20.'),
        parameterSets = cms.vstring('pythia8_example02')
    )
)
process.toTest = ExternalGeneratorFilter(_pythia8)

process.add_(cms.Service('Tracer'))

process.moduleToTest(process.toTest)
)_"};

  edm::test::TestProcessor::Config config{baseConfig};
  SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  SECTION("No event data") {
    edm::test::TestProcessor tester(config);

    REQUIRE_NOTHROW(tester.test());
  }

  SECTION("beginJob and endJob only") {
    edm::test::TestProcessor tester(config);

    REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  }

  SECTION("Run with no LuminosityBlocks") {
    edm::test::TestProcessor tester(config);

    REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
  }

  SECTION("LuminosityBlock with no Events") {
    edm::test::TestProcessor tester(config);

    REQUIRE_NOTHROW(tester.testLuminosityBlockWithNoEvents());
  }
}

//Add additional TEST_CASEs to exercise the modules capabilities
