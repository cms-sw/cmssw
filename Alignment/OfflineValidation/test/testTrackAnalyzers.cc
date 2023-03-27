#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("GeneralPurposeTrackAnalyzer tests", "[GeneralPurposeTrackAnalyzer]") {
  //The python configuration
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
from Alignment.OfflineValidation.generalPurposeTrackAnalyzer_cfi import generalPurposeTrackAnalyzer
process = TestProcess()
process.trackAnalyzer = generalPurposeTrackAnalyzer
process.moduleToTest(process.trackAnalyzer)
process.add_(cms.Service('MessageLogger'))
process.add_(cms.Service('JobReportService'))
process.add_(cms.Service('TFileService',fileName=cms.string('tesTrackAnalyzer1.root')))
)_"};

  edm::test::TestProcessor::Config config{baseConfig};
  SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  // SECTION("No event data") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.test());
  // }

  SECTION("beginJob and endJob only") {
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  }

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
from Alignment.OfflineValidation.dmrChecker_cfi import dmrChecker
process = TestProcess()
process.dmrAnalyzer = dmrChecker
process.moduleToTest(process.dmrAnalyzer)
process.add_(cms.Service('MessageLogger'))
process.add_(cms.Service('JobReportService'))
process.add_(cms.Service('TFileService',fileName=cms.string('tesTrackAnalyzer2.root')))
)_"};

  edm::test::TestProcessor::Config config{baseConfig};
  SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  // SECTION("No event data") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.test());
  // }

  SECTION("beginJob and endJob only") {
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  }

  // SECTION("Run with no LuminosityBlocks") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
  // }

  // SECTION("LuminosityBlock with no Events") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.testLuminosityBlockWithNoEvents());
  // }
}

TEST_CASE("JetHTAnalyzer tests", "[JetHTAnalyzer]") {
  //The python configuration
  edm::test::TestProcessor::Config config{
      R"_(import FWCore.ParameterSet.Config as cms
from FWCore.TestProcessor.TestProcess import *
from Alignment.OfflineValidation.jetHTAnalyzer_cfi import jetHTAnalyzer
process = TestProcess()
process.JetHTAnalyzer = jetHTAnalyzer
process.moduleToTest(process.JetHTAnalyzer)
process.add_(cms.Service('JobReportService'))
process.add_(cms.Service('TFileService',fileName=cms.string('tesTrackAnalyzer3.root')))
)_"};

  SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  SECTION("beginJob and endJob only") {
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  }

  // SECTION("No event data") {
  //  edm::test::TestProcessor tester(config);
  //  REQUIRE_NOTHROW(tester.test());
  //}
}
