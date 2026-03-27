#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <format>
#include <iostream>
#include "TROOT.h"

#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

//Function to run the catch2 tests
//___________________________________________________________________________________________
void runTestForAnalyzer(const std::string& baseConfig, const std::string& analyzerName) {
  edm::test::TestProcessor::Config config{baseConfig};

  SECTION(analyzerName + " base configuration is OK") {
    gROOT->GetList()->Delete();  // to get rid of duplicated ME warnings
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(edm::test::TestProcessor(config));
  }

  SECTION(analyzerName + " No Runs data") {
    gROOT->GetList()->Delete();  // to get rid of duplicated ME warnings
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testWithNoRuns());
  }

  SECTION(analyzerName + " beginJob and endJob only") {
    gROOT->GetList()->Delete();  // to get rid of duplicated ME warnings
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  }

  SECTION(analyzerName + " No event data") {
    gROOT->GetList()->Delete();  // to get rid of duplicated ME warnings
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.test());
  }

  SECTION(analyzerName + " Run with no LuminosityBlocks") {
    gROOT->GetList()->Delete();  // to get rid of duplicated ME warnings
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
  }

  SECTION(analyzerName + " LuminosityBlock with no Events") {
    gROOT->GetList()->Delete();  // to get rid of duplicated ME warnings
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testLuminosityBlockWithNoEvents());
  }
}

// Function to generate base configuration string
//___________________________________________________________________________________________
std::string generateBaseConfig(const std::string& analyzerName) {
  // Define a raw string literal
  constexpr const char* rawString = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("MagneticField.Engine.uniformMagneticField_cfi")
process.load("Configuration.Geometry.GeometryExtended2024Reco_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2024_design', '')
from HLTriggerOffline.Scouting.{}_cfi import {}
process.dqmAnalyzer = {}
process.moduleToTest(process.dqmAnalyzer)
process.add_(cms.Service('DQMStore'))
process.add_(cms.Service('MessageLogger'))
process.add_(cms.Service('JobReportService'))
process.add_(cms.Service('SiteLocalConfigService'))
    )_";

  // Format the raw string literal using std::format
  return std::format(rawString, analyzerName, analyzerName, analyzerName);
}

//___________________________________________________________________________________________
TEST_CASE("ScoutingDileptonMonitor tests", "[ScoutingDileptonMonitor]") {
  const std::string baseConfig = generateBaseConfig("scoutingDileptonMonitor");
  runTestForAnalyzer(baseConfig, "ScoutingDileptonMonitor");
}

//___________________________________________________________________________________________
TEST_CASE("ScoutingPi0Analyzer tests", "[ScoutingPi0Analyzer]") {
  const std::string baseConfig = generateBaseConfig("scoutingPi0Analyzer");
  runTestForAnalyzer(baseConfig, "ScoutingPi0Analyzer");
}

//___________________________________________________________________________________________
TEST_CASE("ElectronEfficiencyPlotter tests", "[ElectronEfficiencyPlotter]") {
  const std::string baseConfig = generateBaseConfig("electronEfficiencyPlotter");
  runTestForAnalyzer(baseConfig, "ElectronEfficiencyPlotter");
}

//___________________________________________________________________________________________
TEST_CASE("PatElectronTagProbeAnalyzer tests", "[PatElectronTagProbeAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("patElectronTagProbeAnalyzer");
  runTestForAnalyzer(baseConfig, "PatElectronTagProbeAnalyzer");
}

//___________________________________________________________________________________________
TEST_CASE("ScoutingEBRecHitAnalyzer tests", "[ScoutingEBRecHitAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("scoutingEBRecHitAnalyzer");
  runTestForAnalyzer(baseConfig, "ScoutingEBRecHitAnalyzer");
}

//___________________________________________________________________________________________
TEST_CASE("ScoutingEGammaCollectionMonitoring tests", "[ScoutingEGammaCollectionMonitoring]") {
  const std::string baseConfig = generateBaseConfig("scoutingEGammaCollectionMonitoring");
  runTestForAnalyzer(baseConfig, "ScoutingEGammaCollectionMonitoring");
}

//___________________________________________________________________________________________
TEST_CASE("ScoutingElectronTagProbeAnalyzer tests", "[ScoutingElectronTagProbeAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("scoutingElectronTagProbeAnalyzer");
  runTestForAnalyzer(baseConfig, "ScoutingElectronTagProbeAnalyzer");
}

//___________________________________________________________________________________________
TEST_CASE("ScoutingHBHERecHitAnalyzer tests", "[ScoutingHBHERecHitAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("scoutingHBHERecHitAnalyzer");
  runTestForAnalyzer(baseConfig, "ScoutingHBHERecHitAnalyzer");
}

//___________________________________________________________________________________________
TEST_CASE("ScoutingMuonPropertiesAnalyzer tests", "[ScoutingMuonPropertiesAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("scoutingMuonPropertiesAnalyzer");
  runTestForAnalyzer(baseConfig, "ScoutingMuonPropertiesAnalyzer");
}

//___________________________________________________________________________________________
TEST_CASE("ScoutingMuonTagProbeAnalyzer tests", "[ScoutingMuonTagProbeAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("scoutingMuonTagProbeAnalyzer");
  runTestForAnalyzer(baseConfig, "ScoutingMuonTagProbeAnalyzer");
}

//___________________________________________________________________________________________
TEST_CASE("ScoutingMuonTriggerAnalyzer tests", "[ScoutingMuonTriggerAnalyzer]") {
  const std::string baseConfig = generateBaseConfig("scoutingMuonTriggerAnalyzer");
  runTestForAnalyzer(baseConfig, "ScoutingMuonTriggerAnalyzer");
}
