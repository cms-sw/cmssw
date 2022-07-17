#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("GeneralPurposeTrackAnalyzer tests", "[GeneralPurposeTrackAnalyzer]") {
  //The python configuration
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
from Alignment.OfflineValidation.GeneralPurposeTrackAnalyzer_cfi import GeneralPurposeTrackAnalyzer
process = TestProcess()
process.trackAnalyzer = GeneralPurposeTrackAnalyzer
process.moduleToTest(process.trackAnalyzer)
process.add_(cms.Service('MessageLogger'))
process.add_(cms.Service('TFileService',fileName=cms.string('tesTrackAnalyzer1.root')))
)_"};

  edm::test::TestProcessor::Config config{baseConfig};
  SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  // SECTION("No event data") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.test());
  // }

  // SECTION("beginJob and endJob only") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  // }

  // SECTION("Run with no LuminosityBlocks") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
  // }

  // SECTION("LuminosityBlock with no Events") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.testLuminosityBlockWithNoEvents());
  // }
}

TEST_CASE("DMRChecker tests", "[DMRChecker]") {
  //The python configuration
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
from Alignment.OfflineValidation.DMRChecker_cfi import DMRChecker
process = TestProcess()
process.dmrAnalyzer = DMRChecker
process.moduleToTest(process.dmrAnalyzer)
process.add_(cms.Service('MessageLogger'))
process.add_(cms.Service('TFileService',fileName=cms.string('tesTrackAnalyzer2.root')))
)_"};

  edm::test::TestProcessor::Config config{baseConfig};
  SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  // SECTION("No event data") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.test());
  // }

  // SECTION("beginJob and endJob only") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  // }

  // SECTION("Run with no LuminosityBlocks") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
  // }

  // SECTION("LuminosityBlock with no Events") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.testLuminosityBlockWithNoEvents());
  // }
}
