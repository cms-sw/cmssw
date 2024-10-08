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

  SECTION("No event data") {
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.test());
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

// Function to generate base configuration string
//___________________________________________________________________________________________
std::string generateBaseConfig(const std::string& analyzerName, const std::string& rootFileName) {
  // Define a raw string literal
  constexpr const char* rawString = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("MagneticField.Engine.uniformMagneticField_cfi")
process.load("Configuration.Geometry.GeometryExtended2024Reco_cff")
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
from DQM.TrackingMonitorSource.{}_cfi import {}
process.trackAnalyzer = {}
process.moduleToTest(process.trackAnalyzer)
process.add_(cms.Service('DQMStore'))
process.add_(cms.Service('MessageLogger'))
process.add_(cms.Service('JobReportService'))
process.add_(cms.Service('TFileService',fileName=cms.string('{}')))
    )_";

  // Format the raw string literal using fmt::format
  return fmt::format(rawString, analyzerName, analyzerName, analyzerName, rootFileName);
}

//___________________________________________________________________________________________
TEST_CASE("ShortenedTrackResolution tests", "[ShortenedTrackResolution]") {
  const std::string baseConfig = generateBaseConfig("shortenedTrackResolution", "test1.root");
  runTestForAnalyzer(baseConfig, "ShortenedTrackResolution");
}

//___________________________________________________________________________________________
TEST_CASE("StandaloneTrackMonitor tests", "[StandaloneTrackMonitor]") {
  const std::string baseConfig = generateBaseConfig("standaloneTrackMonitorDefault", "test2.root");
  runTestForAnalyzer(baseConfig, "StandaloneTrackMonitor");
}

//___________________________________________________________________________________________
TEST_CASE("AlcaRecoTrackSelector tests", "[AlcaRecoTrackSelector]") {
  const std::string baseConfig = generateBaseConfig("alcaRecoTrackSelector", "tes3.root");
  runTestForAnalyzer(baseConfig, "AlcaRecoTrackSelector");
}

//___________________________________________________________________________________________
//TEST_CASE("HltPathSelector tests", "[HltPathSelector]") {
//  const std::string baseConfig = generateBaseConfig("hltPathSelector", "test_hltPathSelector.root");
//  runTestForAnalyzer(baseConfig, "HltPathSelector");
//}

//___________________________________________________________________________________________
TEST_CASE("TrackMultiplicityFilter tests", "[TrackMultiplicityFilter]") {
  const std::string baseConfig = generateBaseConfig("trackMultiplicityFilter", "test_trackMultiplicityFilter.root");
  runTestForAnalyzer(baseConfig, "TrackMultiplicityFilter");
}

//___________________________________________________________________________________________
//TEST_CASE("TrackToTrackComparisonHists tests", "[TrackToTrackComparisonHists]") {
//  const std::string baseConfig = generateBaseConfig("trackToTrackComparisonHists", "test_trackToTrackComparisonHists.root");
//  runTestForAnalyzer(baseConfig, "TrackToTrackComparisonHists");
//}

//___________________________________________________________________________________________
TEST_CASE("TrackTypeMonitor tests", "[TrackTypeMonitor]") {
  const std::string baseConfig = generateBaseConfig("trackTypeMonitor", "test_trackTypeMonitor.root");
  runTestForAnalyzer(baseConfig, "TrackTypeMonitor");
}

//___________________________________________________________________________________________
TEST_CASE("TtbarEventSelector tests", "[TtbarEventSelector]") {
  const std::string baseConfig = generateBaseConfig("ttbarEventSelector", "test_ttbarEventSelector.root");
  runTestForAnalyzer(baseConfig, "TtbarEventSelector");
}

//___________________________________________________________________________________________
TEST_CASE("TtbarTrackProducer tests", "[TtbarTrackProducer]") {
  const std::string baseConfig = generateBaseConfig("ttbarTrackProducer", "test_ttbarTrackProducer.root");
  runTestForAnalyzer(baseConfig, "TtbarTrackProducer");
}

//___________________________________________________________________________________________
TEST_CASE("V0EventSelector tests", "[V0EventSelector]") {
  const std::string baseConfig = generateBaseConfig("v0EventSelector", "test_v0EventSelector.root");
  runTestForAnalyzer(baseConfig, "V0EventSelector");
}

//___________________________________________________________________________________________
TEST_CASE("V0VertexTrackProducer tests", "[V0VertexTrackProducer]") {
  const std::string baseConfig = generateBaseConfig("v0VertexTrackProducer", "test_v0VertexTrackProducer.root");
  runTestForAnalyzer(baseConfig, "V0VertexTrackProducer");
}

//___________________________________________________________________________________________
TEST_CASE("WtoLNuSelector tests", "[WtoLNuSelector]") {
  const std::string baseConfig = generateBaseConfig("wtoLNuSelector", "test_wtoLNuSelector.root");
  runTestForAnalyzer(baseConfig, "WtoLNuSelector");
}

//___________________________________________________________________________________________
TEST_CASE("ZeeDetails tests", "[ZeeDetails]") {
  const std::string baseConfig = generateBaseConfig("zeeDetails", "test_zeeDetails.root");
  runTestForAnalyzer(baseConfig, "ZeeDetails");
}

//___________________________________________________________________________________________
TEST_CASE("ZtoEEElectronTrackProducer tests", "[ZtoEEElectronTrackProducer]") {
  const std::string baseConfig =
      generateBaseConfig("ztoEEElectronTrackProducer", "test_ztoEEElectronTrackProducer.root");
  runTestForAnalyzer(baseConfig, "ZtoEEElectronTrackProducer");
}

//___________________________________________________________________________________________
TEST_CASE("ZtoEEEventSelector tests", "[ZtoEEEventSelector]") {
  const std::string baseConfig = generateBaseConfig("ztoEEEventSelector", "test_ztoEEEventSelector.root");
  runTestForAnalyzer(baseConfig, "ZtoEEEventSelector");
}

//___________________________________________________________________________________________
TEST_CASE("ZtoMMEventSelector tests", "[ZtoMMEventSelector]") {
  const std::string baseConfig = generateBaseConfig("ztoMMEventSelector", "test_ztoMMEventSelector.root");
  runTestForAnalyzer(baseConfig, "ZtoMMEventSelector");
}

//___________________________________________________________________________________________
TEST_CASE("ZtoMMMuonTrackProducer tests", "[ZtoMMMuonTrackProducer]") {
  const std::string baseConfig = generateBaseConfig("ztoMMMuonTrackProducer", "test_ztoMMMuonTrackProducer.root");
  runTestForAnalyzer(baseConfig, "ZtoMMMuonTrackProducer");
}
