#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// Function to run the catch2 tests
//___________________________________________________________________________________________
void runTestForAnalyzer(const std::string& baseConfig, const std::string& analyzerName) {
  edm::test::TestProcessor::Config config{baseConfig};

  SECTION(analyzerName + " base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  SECTION(analyzerName + " No Runs data") {
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testWithNoRuns());
  }

  SECTION(analyzerName + " beginJob and endJob only") {
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  }

  // the following part is commented because the
  // HIPTwoBodyDecayAnalyzer crashes on
  // No "TransientTrackRecord" record found in the EventSetup.

  /* 
  SECTION("No event data") {
   edm::test::TestProcessor tester(config);
   REQUIRE_NOTHROW(tester.test());
  }
  */

  SECTION("Run with no LuminosityBlocks") {
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
  }

  SECTION("LuminosityBlock with no Events") {
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testLuminosityBlockWithNoEvents());
  }
}

// Function to generate base configuration string
//___________________________________________________________________________________________
std::string generateBaseConfig(const std::string& analyzerName, const std::string& rootFileName) {
  // Define a raw string literal
  constexpr const char* rawString = R"_(from FWCore.TestProcessor.TestProcess import *
from Alignment.HIPAlignmentAlgorithm.{}_cfi import {}
process = TestProcess()
process.trackAnalyzer = {}
process.moduleToTest(process.trackAnalyzer)
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.add_(cms.ESProducer("TransientTrackBuilderESProducer"))
process.add_(cms.Service('MessageLogger'))
process.add_(cms.Service('JobReportService'))
process.add_(cms.Service('TFileService',fileName=cms.string('{}')))
    )_";

  // Format the raw string literal using fmt::format
  return fmt::format(rawString, analyzerName, analyzerName, analyzerName, rootFileName);
}

//___________________________________________________________________________________________
TEST_CASE("LhcTrackAnalyzer tests", "[LhcTrackAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("lhcTrackAnalyzer", "testHIPAnalyzers1.root");
  runTestForAnalyzer(baseConfig, "LhcTrackAnalyzer");
}

//___________________________________________________________________________________________
TEST_CASE("HIPTwoBodyDecayAnalyzer tests", "[HIPTwoBodyDecayAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("hipTwoBodyDecayAnalyzer", "testHIPAnalyzers2.root");
  runTestForAnalyzer(baseConfig, "HIPTwoBodyDecayAnalyzer");
}
-- dummy change --
