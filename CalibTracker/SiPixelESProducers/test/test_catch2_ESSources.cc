#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// Function to run the catch2 tests
//___________________________________________________________________________________________
void runTestForEsProducer(const std::string& baseConfig, const std::string& esProducerName) {
  edm::test::TestProcessor::Config config{baseConfig};

  SECTION(esProducerName + " base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  SECTION(esProducerName + " No Runs data") {
    edm::test::TestProcessor tester(config);
    REQUIRE_NOTHROW(tester.testWithNoRuns());
  }

  SECTION(esProducerName + " beginJob and endJob only") {
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
std::string generateBaseConfig(const std::string& esProducerName,
                               const std::string& recordName,
                               const std::string& dataName) {
  // Define a raw string literal
  constexpr const char* rawString = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("MagneticField.Engine.uniformMagneticField_cfi")
process.load("Configuration.Geometry.GeometryExtended2024Reco_cff")
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.getConditions = cms.EDAnalyzer( "EventSetupRecordDataGetter",
 toGet = cms.VPSet(cms.PSet(
        record = cms.string("{}"),
        data = cms.vstring("{}")
        )),
    verbose = cms.untracked.bool(True)
)
process.add_(cms.ESSource("{}"))
process.add_(cms.Service('MessageLogger'))
process.add_(cms.Service('JobReportService'))
process.moduleToTest(process.getConditions)
    )_";

  // Format the raw string literal using fmt::format
  return fmt::format(rawString, recordName, dataName, esProducerName);
}

//___________________________________________________________________________________________
TEST_CASE("SiPixelFakeGenErrorDBObjectESSource tests", "[SiPixelFakeGenErrorDBObjectESSource]") {
  const std::string baseConfig = generateBaseConfig(
      "SiPixelFakeGenErrorDBObjectESSource", "SiPixelGenErrorDBObjectRcd", "SiPixelGenErrorDBObject");
  runTestForEsProducer(baseConfig, "SiPixelFakeGenErrorDBObjectESSource");
}

//___________________________________________________________________________________________
TEST_CASE("SiPixelFakeTemplateDBObjectESSource tests", "[SiPixelFakeTemplateDBObjectESSource]") {
  const std::string baseConfig = generateBaseConfig(
      "SiPixelFakeTemplateDBObjectESSource", "SiPixelTemplateDBObjectRcd", "SiPixelTemplateDBObject");
  runTestForEsProducer(baseConfig, "SiPixelFakeTemplateDBObjectESSource");
}

//___________________________________________________________________________________________
TEST_CASE("SiPixelFakeLorentzAngleESSource tests", "[SiPixelFakeLorentzAngleESSource]") {
  const std::string baseConfig =
      generateBaseConfig("SiPixelFakeLorentzAngleESSource", "SiPixelLorentzAngleRcd", "SiPixelLorentzAngle");
  runTestForEsProducer(baseConfig, "SiPixelFakeLorentzAngleESSource");
}

//___________________________________________________________________________________________
TEST_CASE("SiPixelFakeQualityESSource tests", "[SiPixelFakeQualityESSource]") {
  const std::string baseConfig =
      generateBaseConfig("SiPixelFakeQualityESSource", "SiPixelQualityFromDbRcd", "SiPixelQuality");
  runTestForEsProducer(baseConfig, "SiPixelFakeQualityESSource");
}

//___________________________________________________________________________________________
TEST_CASE("SiPixelFakeGainESSource tests", "[SiPixelFakeGainESSource]") {
  const std::string baseConfig =
      generateBaseConfig("SiPixelFakeGainESSource", "SiPixelGainCalibrationRcd", "SiPixelGainCalibration");
  runTestForEsProducer(baseConfig, "SiPixelFakeGainESSource");
}

//___________________________________________________________________________________________
TEST_CASE("SiPixelFakeGainForHLTESSource tests", "[SiPixelFakeGainForHLTESSource]") {
  const std::string baseConfig = generateBaseConfig(
      "SiPixelFakeGainForHLTESSource", "SiPixelGainCalibrationForHLTRcd", "SiPixelGainCalibrationForHLT");
  runTestForEsProducer(baseConfig, "SiPixelFakeGainForHLTESSource");
}

//___________________________________________________________________________________________
TEST_CASE("SiPixelFakeGainOfflineESSource tests", "[SiPixelFakeGainOfflineESSource]") {
  const std::string baseConfig = generateBaseConfig(
      "SiPixelFakeGainOfflineESSource", "SiPixelGainCalibrationOfflineRcd", "SiPixelGainCalibrationOffline");
  runTestForEsProducer(baseConfig, "SiPixelFakeGainOfflineESSource");
}
