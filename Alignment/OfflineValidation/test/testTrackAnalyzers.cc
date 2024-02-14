#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Alignment/OfflineValidation/interface/TkAlStyle.h"

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

  // Add more sections as needed

  //SECTION("No event data") {
  //  edm::test::TestProcessor tester(config);
  //  REQUIRE_NOTHROW(tester.test());
  //}

  // SECTION("Run with no LuminosityBlocks") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
  // }

  // SECTION("LuminosityBlock with no Events") {
  //   edm::test::TestProcessor tester(config);
  //   REQUIRE_NOTHROW(tester.testLuminosityBlockWithNoEvents());
  // }
}

// Function to generate base configuration string
//___________________________________________________________________________________________
std::string generateBaseConfig(const std::string& analyzerName, const std::string& rootFileName) {
  // Define a raw string literal
  constexpr const char* rawString = R"_(from FWCore.TestProcessor.TestProcess import *
from Alignment.OfflineValidation.{}_cfi import {}
process = TestProcess()
process.trackAnalyzer = {}
process.moduleToTest(process.trackAnalyzer)
process.add_(cms.Service('MessageLogger'))
process.add_(cms.Service('JobReportService'))
process.add_(cms.Service('TFileService',fileName=cms.string('{}')))
    )_";

  // Format the raw string literal using fmt::format
  return fmt::format(rawString, analyzerName, analyzerName, analyzerName, rootFileName);
}

//___________________________________________________________________________________________
TEST_CASE("GeneralPurposeTrackAnalyzer tests", "[GeneralPurposeTrackAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("generalPurposeTrackAnalyzer", "tesTrackAnalyzer0.root");
  runTestForAnalyzer(baseConfig, "GeneralPurposeTrackAnalyzer");
}

//___________________________________________________________________________________________
TEST_CASE("GeneralPurposeVertexAnalyzer tests", "[GeneralPurposeVertexAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("generalPurposeVertexAnalyzer", "tesVertexAnalyzer1.root");
  runTestForAnalyzer(baseConfig, "GeneralPurposeVertexAnalyzer");
}

//___________________________________________________________________________________________
TEST_CASE("DMRChecker tests", "[DMRChecker]") {
  const std::string baseConfig = generateBaseConfig("dmrChecker", "tesTrackAnalyzer2.root");
  runTestForAnalyzer(baseConfig, "DMRChecker");
}

//___________________________________________________________________________________________
TEST_CASE("JetHTAnalyzer tests", "[JetHTAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("jetHTAnalyzer", "tesTrackAnalyzer3.root");
  runTestForAnalyzer(baseConfig, "JetHTAnalyzer");
}

//___________________________________________________________________________________________
TEST_CASE("DiMuonValidation tests", "[DiMuonValidation]") {
  const std::string baseConfig = generateBaseConfig("diMuonValidation", "tesTrackAnalyzer4.root");
  runTestForAnalyzer(baseConfig, "DiMuonValidation");
}

//___________________________________________________________________________________________
TEST_CASE("CosmicSplitterValidation tests", "[CosmicsSplitterValidation]") {
  const std::string baseConfig = generateBaseConfig("cosmicSplitterValidation", "tesTrackAnalyzer5.root");
  runTestForAnalyzer(baseConfig, "CosmicSplitterValidation");
}

//___________________________________________________________________________________________
TEST_CASE("DiElectronVertexValidation tests", "[DiElectronVertexValidation]") {
  const std::string baseConfig = generateBaseConfig("diElectronVertexValidation", "tesTrackAnalyzer6.root");
  runTestForAnalyzer(baseConfig, "DiElectronVertexValidation");
}

//___________________________________________________________________________________________
TEST_CASE("DiMuonVertexValidation tests", "[DiMuonVertexValidation]") {
  const std::string baseConfig = generateBaseConfig("diMuonVertexValidation", "tesTrackAnalyzer7.root");
  runTestForAnalyzer(baseConfig, "DiMuonVertexValidation");
}

//___________________________________________________________________________________________
TEST_CASE("EopElecTreeWriter tests", "[EopElecTreeWriter]") {
  const std::string baseConfig = generateBaseConfig("eopElecTreeWriter", "tesTrackAnalyzer8.root");
  runTestForAnalyzer(baseConfig, "EopElecTreeWriter");
}

//___________________________________________________________________________________________
TEST_CASE("EopTreeWriter tests", "[EopTreeWriter]") {
  const std::string baseConfig = generateBaseConfig("eopTreeWriter", "tesTrackAnalyzer9.root");
  runTestForAnalyzer(baseConfig, "EopTreeWriter");
}

//___________________________________________________________________________________________
TEST_CASE("OverlapValidation tests", "[OverlapValidation]") {
  const std::string baseConfig = generateBaseConfig("overlapValidation", "tesTrackAnalyzer10.root");
  runTestForAnalyzer(baseConfig, "OverlapValidation");
}

//___________________________________________________________________________________________
TEST_CASE("PixelBaryCentreAnalyzer tests", "[PixelBaryCentreAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("pixelBaryCentreAnalyzer", "tesTrackAnalyzer11.root");
  runTestForAnalyzer(baseConfig, "PixelBaryCentreAnalyzer");
}

//___________________________________________________________________________________________
TEST_CASE("PrimaryVertexValidation tests", "[PrimaryVertexValidation]") {
  const std::string baseConfig = generateBaseConfig("primaryVertexValidation", "tesTrackAnalyzer12.root");
  runTestForAnalyzer(baseConfig, "PrimaryVertexValidation");
}

//___________________________________________________________________________________________
TEST_CASE("SplitVertexResolution tests", "[SplitVertexResolution]") {
  const std::string baseConfig = generateBaseConfig("splitVertexResolution", "tesTrackAnalyzer13.root");
  runTestForAnalyzer(baseConfig, "SplitVertexResolution");
}

//___________________________________________________________________________________________
TEST_CASE("TrackerGeometryIntoNtuples tests", "[TrackerGeometryIntoNtuples]") {
  const std::string baseConfig = generateBaseConfig("trackerGeometryIntoNtuples", "tesTrackAnalyzer14.root");
  runTestForAnalyzer(baseConfig, "TrackerGeometryIntoNtuples");
}

//___________________________________________________________________________________________
TEST_CASE("TrackerOfflineValidation tests", "[TrackerOfflineValidation]") {
  const std::string baseConfig = generateBaseConfig("TrackerOfflineValidation", "tesTrackAnalyzer15.root");
  runTestForAnalyzer(baseConfig, "TrackerOfflineValidation");
}

//___________________________________________________________________________________________
TEST_CASE("TrackerGeometryCompare tests", "[TrackerGeometryCompare]") {
  const std::string baseConfig = generateBaseConfig("trackerGeometryCompare", "tesTrackAnalyzer16.root");
  runTestForAnalyzer(baseConfig, "trackerGeometryCompare");
}
