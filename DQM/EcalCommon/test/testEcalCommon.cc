#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// Function to run the catch2 tests
//___________________________________________________________________________________________
void runTestForAnalyzer(const std::string& baseConfig, const std::string& analyzerName) {
  edm::test::TestProcessor::Config config{baseConfig};

  SECTION(analyzerName + " base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  SECTION("Run with no LuminosityBlocks") {
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
  }
}

// Function to generate base configuration string
//___________________________________________________________________________________________
std::string generateBaseConfig(const std::string& cfiName, const std::string& analyzerName) {
  // Define a raw string literal
  constexpr const char* rawString = R"_(from FWCore.TestProcessor.TestProcess import *
from DQM.EcalCommon.{}_cfi import {}
process = TestProcess()
process.harvester = {}
process.moduleToTest(process.harvester)
process.add_(cms.Service('MessageLogger'))
process.add_(cms.Service('JobReportService'))
process.add_(cms.Service('DQMStore'))
    )_";

  // Format the raw string literal using fmt::format
  return fmt::format(rawString, cfiName, analyzerName, analyzerName);
}

//___________________________________________________________________________________________
TEST_CASE("EcalMEFormatter tests", "[EcalMEFormatter]") {
  const std::string baseConfig = generateBaseConfig("EcalMEFormatter", "ecalMEFormatter");
  runTestForAnalyzer(baseConfig, "EcalMEFormatter");
}
