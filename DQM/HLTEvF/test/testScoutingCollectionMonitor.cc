#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <format>
#include <iostream>
#include "TROOT.h"

// Scouting data formats needed to put mock collections
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"
#include "DataFormats/Scouting/interface/Run3ScoutingParticle.h"
#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "DataFormats/Common/interface/ValueMap.h"

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
from DQM.HLTEvF.{}_cfi import {}
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

// Generates a config where the beamspot label is replaced with a label
// that will never be produced, exercising the invalid-beamspot branch.
// All mandatory collections are declared as "to be put" by the test harness.
//___________________________________________________________________________________________
std::string generateNoBeamspotConfig(const std::string& analyzerName) {
  constexpr const char* rawString = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("MagneticField.Engine.uniformMagneticField_cfi")
process.load("Configuration.Geometry.GeometryExtended2024Reco_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2024_design', '')
from DQM.HLTEvF.{}_cfi import {}
process.dqmAnalyzer = {}
# Override the beamSpot to point at a label that will never be produced,
# so getValidHandle returns false for beamspot while all mandatory
# collections are still present (supplied via TestProcessor::put calls).
process.dqmAnalyzer.beamSpot = cms.InputTag("nonExistentBeamSpot")
process.dqmAnalyzer.onlyScouting = cms.bool(True)
process.moduleToTest(process.dqmAnalyzer)
process.add_(cms.Service('DQMStore'))
process.add_(cms.Service('MessageLogger'))
process.add_(cms.Service('JobReportService'))
process.add_(cms.Service('SiteLocalConfigService'))
    )_";

  return std::format(rawString, analyzerName, analyzerName, analyzerName);
}

//___________________________________________________________________________________________
TEST_CASE("ScoutingCollectionMonitor tests", "[ScoutingCollectionMonitor]") {
  const std::string baseConfig = generateBaseConfig("scoutingCollectionMonitor");
  runTestForAnalyzer(baseConfig, "ScoutingCollectionMonitor");
}

// Separate test case that exercises the invalid-beamspot code path.
// All mandatory collections are supplied as empty vectors so the early-return
// guard in analyze() is satisfied, but the beamspot handle will be invalid,
// so the beamspot-based histogram fills (tk_BS_dxy, tk_BS_dz, trkd0BS, trkdzBS)
// are skipped, verifying that branch does not crash.
//___________________________________________________________________________________________
TEST_CASE("ScoutingCollectionMonitor invalid beamspot", "[ScoutingCollectionMonitor]") {
  const std::string config = generateNoBeamspotConfig("scoutingCollectionMonitor");
  edm::test::TestProcessor::Config cfg{config};

  // The first argument must match the module label used in the InputTag
  // in the Python config (e.g. cms.InputTag("hltScoutingMuonPackerNoVtx"))
  auto muonsToken = cfg.produces<std::vector<Run3ScoutingMuon>>("hltScoutingMuonPackerNoVtx");
  auto muonsVtxToken = cfg.produces<std::vector<Run3ScoutingMuon>>("hltScoutingMuonPackerVtx");
  auto electronsToken = cfg.produces<std::vector<Run3ScoutingElectron>>("hltScoutingEgammaPacker");
  auto photonsToken = cfg.produces<std::vector<Run3ScoutingPhoton>>("hltScoutingEgammaPacker");
  auto pfjetsToken = cfg.produces<std::vector<Run3ScoutingPFJet>>("hltScoutingPFPacker");
  auto pfcandsToken = cfg.produces<std::vector<Run3ScoutingParticle>>("hltScoutingPFPacker");
  auto tracksToken = cfg.produces<std::vector<Run3ScoutingTrack>>("hltScoutingTrackPacker");
  auto pvToken = cfg.produces<std::vector<Run3ScoutingVertex>>("hltScoutingPrimaryVertexPacker", "primaryVtx");
  auto dvToken = cfg.produces<std::vector<Run3ScoutingVertex>>("hltScoutingMuonPackerVtx", "displacedVtx");
  auto dvNoVtxToken = cfg.produces<std::vector<Run3ScoutingVertex>>("hltScoutingMuonPackerNoVtx", "displacedVtx");
  auto rhoToken = cfg.produces<double>("hltScoutingPFPacker", "rho");
  auto metPtToken = cfg.produces<double>("hltScoutingPFPacker", "pfMetPt");
  auto metPhiToken = cfg.produces<double>("hltScoutingPFPacker", "pfMetPhi");

  const std::string prod = "run3ScoutingElectronBestTrack";
  auto vmBestTrackIndexToken = cfg.produces<edm::ValueMap<int>>(prod, "Run3ScoutingElectronBestTrackIndex");
  auto vmTrkd0Token = cfg.produces<edm::ValueMap<float>>(prod, "Run3ScoutingElectronTrackd0");
  auto vmTrkdzToken = cfg.produces<edm::ValueMap<float>>(prod, "Run3ScoutingElectronTrackdz");
  auto vmTrkptToken = cfg.produces<edm::ValueMap<float>>(prod, "Run3ScoutingElectronTrackpt");
  auto vmTrketaToken = cfg.produces<edm::ValueMap<float>>(prod, "Run3ScoutingElectronTracketa");
  auto vmTrkphiToken = cfg.produces<edm::ValueMap<float>>(prod, "Run3ScoutingElectronTrackphi");
  auto vmTrkpModeToken = cfg.produces<edm::ValueMap<float>>(prod, "Run3ScoutingElectronTrackpMode");
  auto vmTrketaModeToken = cfg.produces<edm::ValueMap<float>>(prod, "Run3ScoutingElectronTracketaMode");
  auto vmTrkphiModeToken = cfg.produces<edm::ValueMap<float>>(prod, "Run3ScoutingElectronTrackphiMode");
  auto vmTrkqoverpModeErrorToken = cfg.produces<edm::ValueMap<float>>(prod, "Run3ScoutingElectronTrackqoverpModeError");
  auto vmTrkchi2overndfToken = cfg.produces<edm::ValueMap<float>>(prod, "Run3ScoutingElectronTrackchi2overndf");
  auto vmTrkchargeToken = cfg.produces<edm::ValueMap<int>>(prod, "Run3ScoutingElectronTrackcharge");

  auto tracks = std::make_unique<std::vector<Run3ScoutingTrack>>();
  tracks->emplace_back();  // default constructed

  SECTION("Runs without crashing when beamspot handle is invalid") {
    gROOT->GetList()->Delete();
    edm::test::TestProcessor tester(cfg);

    REQUIRE_NOTHROW(tester.test(std::make_pair(muonsToken, std::make_unique<std::vector<Run3ScoutingMuon>>()),
                                std::make_pair(muonsVtxToken, std::make_unique<std::vector<Run3ScoutingMuon>>()),
                                std::make_pair(electronsToken, std::make_unique<std::vector<Run3ScoutingElectron>>()),
                                std::make_pair(photonsToken, std::make_unique<std::vector<Run3ScoutingPhoton>>()),
                                std::make_pair(pfjetsToken, std::make_unique<std::vector<Run3ScoutingPFJet>>()),
                                std::make_pair(pfcandsToken, std::make_unique<std::vector<Run3ScoutingParticle>>()),
                                std::make_pair(tracksToken, std::move(tracks)),
                                std::make_pair(pvToken, std::make_unique<std::vector<Run3ScoutingVertex>>()),
                                std::make_pair(dvToken, std::make_unique<std::vector<Run3ScoutingVertex>>()),
                                std::make_pair(dvNoVtxToken, std::make_unique<std::vector<Run3ScoutingVertex>>()),
                                std::make_pair(rhoToken, std::make_unique<double>(0.0)),
                                std::make_pair(metPtToken, std::make_unique<double>(0.0)),
                                std::make_pair(metPhiToken, std::make_unique<double>(0.0)),
                                std::make_pair(vmBestTrackIndexToken, std::make_unique<edm::ValueMap<int>>()),
                                std::make_pair(vmTrkd0Token, std::make_unique<edm::ValueMap<float>>()),
                                std::make_pair(vmTrkdzToken, std::make_unique<edm::ValueMap<float>>()),
                                std::make_pair(vmTrkptToken, std::make_unique<edm::ValueMap<float>>()),
                                std::make_pair(vmTrketaToken, std::make_unique<edm::ValueMap<float>>()),
                                std::make_pair(vmTrkphiToken, std::make_unique<edm::ValueMap<float>>()),
                                std::make_pair(vmTrkpModeToken, std::make_unique<edm::ValueMap<float>>()),
                                std::make_pair(vmTrketaModeToken, std::make_unique<edm::ValueMap<float>>()),
                                std::make_pair(vmTrkphiModeToken, std::make_unique<edm::ValueMap<float>>()),
                                std::make_pair(vmTrkqoverpModeErrorToken, std::make_unique<edm::ValueMap<float>>()),
                                std::make_pair(vmTrkchi2overndfToken, std::make_unique<edm::ValueMap<float>>()),
                                std::make_pair(vmTrkchargeToken, std::make_unique<edm::ValueMap<int>>())));
  }
}
