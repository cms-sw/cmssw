#include "catch.hpp"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

static constexpr auto s_tag = "[ExternalLHEProducer]";

TEST_CASE("Standard checks of ExternalLHEProducer", s_tag) {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.externalLHEProducer = cms.EDProducer('ExternalLHEProducer',
    scriptName = cms.FileInPath("GeneratorInterface/LHEInterface/test/run_dummy_script.sh"),
    outputFile = cms.string("dummy.lhe"),
    numberOfParameters = cms.uint32(1),
    args = cms.vstring('value'),
    nEvents = cms.untracked.uint32(5),
    storeXML = cms.untracked.bool(False),
    generateConcurrently = cms.untracked.bool(False)
 )
process.moduleToTest(process.externalLHEProducer)
process.RandomNumberGeneratorService = cms.Service('RandomNumberGeneratorService',
     externalLHEProducer = cms.PSet(
        initialSeed = cms.untracked.uint32(563)
    )
)

)_"};

  edm::test::TestProcessor::Config config{baseConfig};
  SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  SECTION("All events") {
    edm::test::TestProcessor tester(config);

    //there are 5 events in the input
    {
      auto event = tester.test();
      auto const& prod = event.get<LHEEventProduct>();
      REQUIRE(prod->hepeup().NUP == 12);
    }
    REQUIRE_NOTHROW(tester.test());
    REQUIRE_NOTHROW(tester.test());
    REQUIRE_NOTHROW(tester.test());
    REQUIRE_NOTHROW(tester.test());
  }

  SECTION("All events: generateConcurrently") {
    edm::test::TestProcessor::Config config{baseConfig + "\nprocess.externalLHEProducer.generateConcurrently = True\n"};
    edm::test::TestProcessor tester(config);

    //there are 5 events in the input
    {
      auto event = tester.test();
      auto const& prod = event.get<LHEEventProduct>();
      REQUIRE(prod->hepeup().NUP == 12);
    }
    REQUIRE_NOTHROW(tester.test());
    REQUIRE_NOTHROW(tester.test());
    REQUIRE_NOTHROW(tester.test());
    REQUIRE_NOTHROW(tester.test());
  }

  SECTION("Missing events") {
    edm::test::TestProcessor tester(config);

    //there are 5 events in the input
    REQUIRE_NOTHROW(tester.test());
    REQUIRE_NOTHROW(tester.test());
    REQUIRE_NOTHROW(tester.test());
    REQUIRE_NOTHROW(tester.test());
    REQUIRE_THROWS_AS(tester.testEndRun(), cms::Exception);
  }

  SECTION("beginJob and endJob only") {
    edm::test::TestProcessor tester(config);

    REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  }

  SECTION("Run with no LuminosityBlocks") {
    edm::test::TestProcessor tester(config);

    REQUIRE_THROWS_AS(tester.testRunWithNoLuminosityBlocks(), cms::Exception);
  }

  SECTION("LuminosityBlock with no Events") {
    edm::test::TestProcessor tester(config);

    REQUIRE_THROWS_AS(tester.testLuminosityBlockWithNoEvents(), cms::Exception);
  }
}

//Add additional TEST_CASEs to exercise the modules capabilities
