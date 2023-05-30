// -*- C++ -*-
//
// Package:    DataFormats/Scouting
// Class:      TestReadRun3Scouting
//
/**\class edmtest::TestReadRun3Scouting
  Description: Used as part of tests that ensure the run 3 Scouting
  data formats can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time a run 3 Scouting persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of these data formats.
*/
// Original Author:  W. David Dagenhart
//         Created:  18 May 2023

#include "DataFormats/Scouting/interface/Run3ScoutingCaloJet.h"
#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
#include "DataFormats/Scouting/interface/Run3ScoutingHitPatternPOD.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingParticle.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"
#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <vector>

namespace edmtest {

  class TestReadRun3Scouting : public edm::global::EDAnalyzer<> {
  public:
    TestReadRun3Scouting(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void analyzeCaloJets(edm::Event const&) const;
    void analyzeElectrons(edm::Event const&) const;
    void analyzeMuons(edm::Event const&) const;
    void analyzeParticles(edm::Event const&) const;
    void analyzePFJets(edm::Event const&) const;
    void analyzePhotons(edm::Event const&) const;
    void analyzeTracks(edm::Event const&) const;
    void analyzeVertexes(edm::Event const&) const;

    void throwWithMessage(const char*) const;

    // These expected values are meaningless other than we use them
    // to check that values read from persistent storage match the values
    // we know were written.

    std::vector<double> expectedCaloJetsValues_;
    edm::EDGetTokenT<std::vector<Run3ScoutingCaloJet>> caloJetsToken_;

    std::vector<double> expectedElectronFloatingPointValues_;
    std::vector<int> expectedElectronIntegralValues_;
    edm::EDGetTokenT<std::vector<Run3ScoutingElectron>> electronsToken_;

    std::vector<double> expectedMuonFloatingPointValues_;
    std::vector<int> expectedMuonIntegralValues_;
    edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> muonsToken_;

    std::vector<double> expectedParticleFloatingPointValues_;
    std::vector<int> expectedParticleIntegralValues_;
    edm::EDGetTokenT<std::vector<Run3ScoutingParticle>> particlesToken_;

    std::vector<double> expectedPFJetFloatingPointValues_;
    std::vector<int> expectedPFJetIntegralValues_;
    edm::EDGetTokenT<std::vector<Run3ScoutingPFJet>> pfJetsToken_;

    std::vector<double> expectedPhotonFloatingPointValues_;
    std::vector<int> expectedPhotonIntegralValues_;
    edm::EDGetTokenT<std::vector<Run3ScoutingPhoton>> photonsToken_;

    std::vector<double> expectedTrackFloatingPointValues_;
    std::vector<int> expectedTrackIntegralValues_;
    edm::EDGetTokenT<std::vector<Run3ScoutingTrack>> tracksToken_;

    std::vector<double> expectedVertexFloatingPointValues_;
    std::vector<int> expectedVertexIntegralValues_;
    edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> vertexesToken_;
  };

  TestReadRun3Scouting::TestReadRun3Scouting(edm::ParameterSet const& iPSet)
      : expectedCaloJetsValues_(iPSet.getParameter<std::vector<double>>("expectedCaloJetsValues")),
        caloJetsToken_(consumes(iPSet.getParameter<edm::InputTag>("caloJetsTag"))),
        expectedElectronFloatingPointValues_(
            iPSet.getParameter<std::vector<double>>("expectedElectronFloatingPointValues")),
        expectedElectronIntegralValues_(iPSet.getParameter<std::vector<int>>("expectedElectronIntegralValues")),
        electronsToken_(consumes(iPSet.getParameter<edm::InputTag>("electronsTag"))),
        expectedMuonFloatingPointValues_(iPSet.getParameter<std::vector<double>>("expectedMuonFloatingPointValues")),
        expectedMuonIntegralValues_(iPSet.getParameter<std::vector<int>>("expectedMuonIntegralValues")),
        muonsToken_(consumes(iPSet.getParameter<edm::InputTag>("muonsTag"))),
        expectedParticleFloatingPointValues_(
            iPSet.getParameter<std::vector<double>>("expectedParticleFloatingPointValues")),
        expectedParticleIntegralValues_(iPSet.getParameter<std::vector<int>>("expectedParticleIntegralValues")),
        particlesToken_(consumes(iPSet.getParameter<edm::InputTag>("particlesTag"))),
        expectedPFJetFloatingPointValues_(iPSet.getParameter<std::vector<double>>("expectedPFJetFloatingPointValues")),
        expectedPFJetIntegralValues_(iPSet.getParameter<std::vector<int>>("expectedPFJetIntegralValues")),
        pfJetsToken_(consumes(iPSet.getParameter<edm::InputTag>("pfJetsTag"))),
        expectedPhotonFloatingPointValues_(
            iPSet.getParameter<std::vector<double>>("expectedPhotonFloatingPointValues")),
        expectedPhotonIntegralValues_(iPSet.getParameter<std::vector<int>>("expectedPhotonIntegralValues")),
        photonsToken_(consumes(iPSet.getParameter<edm::InputTag>("photonsTag"))),
        expectedTrackFloatingPointValues_(iPSet.getParameter<std::vector<double>>("expectedTrackFloatingPointValues")),
        expectedTrackIntegralValues_(iPSet.getParameter<std::vector<int>>("expectedTrackIntegralValues")),
        tracksToken_(consumes(iPSet.getParameter<edm::InputTag>("tracksTag"))),
        expectedVertexFloatingPointValues_(
            iPSet.getParameter<std::vector<double>>("expectedVertexFloatingPointValues")),
        expectedVertexIntegralValues_(iPSet.getParameter<std::vector<int>>("expectedVertexIntegralValues")),
        vertexesToken_(consumes(iPSet.getParameter<edm::InputTag>("vertexesTag"))) {}

  void TestReadRun3Scouting::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    analyzeCaloJets(iEvent);
    analyzeElectrons(iEvent);
    analyzeMuons(iEvent);
    analyzeParticles(iEvent);
    analyzePFJets(iEvent);
    analyzePhotons(iEvent);
    analyzeTracks(iEvent);
    analyzeVertexes(iEvent);
  }

  void TestReadRun3Scouting::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<double>>("expectedCaloJetsValues");
    desc.add<edm::InputTag>("caloJetsTag");
    desc.add<std::vector<double>>("expectedElectronFloatingPointValues");
    desc.add<std::vector<int>>("expectedElectronIntegralValues");
    desc.add<edm::InputTag>("electronsTag");
    desc.add<std::vector<double>>("expectedMuonFloatingPointValues");
    desc.add<std::vector<int>>("expectedMuonIntegralValues");
    desc.add<edm::InputTag>("muonsTag");
    desc.add<std::vector<double>>("expectedParticleFloatingPointValues");
    desc.add<std::vector<int>>("expectedParticleIntegralValues");
    desc.add<edm::InputTag>("particlesTag");
    desc.add<std::vector<double>>("expectedPFJetFloatingPointValues");
    desc.add<std::vector<int>>("expectedPFJetIntegralValues");
    desc.add<edm::InputTag>("pfJetsTag");
    desc.add<std::vector<double>>("expectedPhotonFloatingPointValues");
    desc.add<std::vector<int>>("expectedPhotonIntegralValues");
    desc.add<edm::InputTag>("photonsTag");
    desc.add<std::vector<double>>("expectedTrackFloatingPointValues");
    desc.add<std::vector<int>>("expectedTrackIntegralValues");
    desc.add<edm::InputTag>("tracksTag");
    desc.add<std::vector<double>>("expectedVertexFloatingPointValues");
    desc.add<std::vector<int>>("expectedVertexIntegralValues");
    desc.add<edm::InputTag>("vertexesTag");
    descriptions.addDefault(desc);
  }

  void TestReadRun3Scouting::analyzeCaloJets(edm::Event const& iEvent) const {
    if (expectedCaloJetsValues_.size() != 16) {
      throwWithMessage("analyzeCaloJets, test configuration error, expectedCaloJetsValues must have size 16");
    }
    auto const& caloJets = iEvent.get(caloJetsToken_);
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    if (caloJets.size() != vectorSize) {
      throwWithMessage("analyzeCaloJets, caloJets does not have expected size");
    }
    unsigned int i = 0;
    for (auto const& caloJet : caloJets) {
      double offset = static_cast<double>(iEvent.id().event() + i);

      if (caloJet.pt() != expectedCaloJetsValues_[0] + offset) {
        throwWithMessage("analyzeCaloJets, pt does not equal expected value");
      }
      if (caloJet.eta() != expectedCaloJetsValues_[1] + offset) {
        throwWithMessage("analyzeCaloJets, eta does not equal expected value");
      }
      if (caloJet.phi() != expectedCaloJetsValues_[2] + offset) {
        throwWithMessage("analyzeCaloJets, phi does not equal expected value");
      }
      if (caloJet.m() != expectedCaloJetsValues_[3] + offset) {
        throwWithMessage("analyzeCaloJets, m does not equal expected value");
      }
      if (caloJet.jetArea() != expectedCaloJetsValues_[4] + offset) {
        throwWithMessage("analyzeCaloJets, jetArea does not equal expected value");
      }
      if (caloJet.maxEInEmTowers() != expectedCaloJetsValues_[5] + offset) {
        throwWithMessage("analyzeCaloJets,  maxEInEmTowers() does not equal expected value");
      }
      if (caloJet.maxEInHadTowers() != expectedCaloJetsValues_[6] + offset) {
        throwWithMessage("analyzeCaloJets,  maxEInHadTowers does not equal expected value");
      }
      if (caloJet.hadEnergyInHB() != expectedCaloJetsValues_[7] + offset) {
        throwWithMessage("analyzeCaloJets, hadEnergyInHB does not equal expected value");
      }
      if (caloJet.hadEnergyInHE() != expectedCaloJetsValues_[8] + offset) {
        throwWithMessage("analyzeCaloJets, hadEnergyInHE does not equal expected value");
      }
      if (caloJet.hadEnergyInHF() != expectedCaloJetsValues_[9] + offset) {
        throwWithMessage("analyzeCaloJets, hadEnergyInHF does not equal expected value");
      }
      if (caloJet.emEnergyInEB() != expectedCaloJetsValues_[10] + offset) {
        throwWithMessage("analyzeCaloJets, emEnergyInEB does not equal expected value");
      }
      if (caloJet.emEnergyInEE() != expectedCaloJetsValues_[11] + offset) {
        throwWithMessage("analyzeCaloJets, emEnergyInEE does not equal expected value");
      }
      if (caloJet.emEnergyInHF() != expectedCaloJetsValues_[12] + offset) {
        throwWithMessage("analyzeCaloJets, emEnergyInHF does not equal expected value");
      }
      if (caloJet.towersArea() != expectedCaloJetsValues_[13] + offset) {
        throwWithMessage("analyzeCaloJets, towersArea does not equal expected value");
      }
      if (caloJet.mvaDiscriminator() != expectedCaloJetsValues_[14] + offset) {
        throwWithMessage("analyzeCaloJets,  mvaDiscriminator does not equal expected value");
      }
      if (caloJet.btagDiscriminator() != expectedCaloJetsValues_[15] + offset) {
        throwWithMessage("analyzeCaloJets,  btagDiscriminator does not equal expected value");
      }
      ++i;
    }
  }

  void TestReadRun3Scouting::analyzeElectrons(edm::Event const& iEvent) const {
    if (expectedElectronFloatingPointValues_.size() != 19) {
      throwWithMessage(
          "analyzeElectrons, test configuration error, expectedElectronFloatingPointValues must have size 19");
    }
    if (expectedElectronIntegralValues_.size() != 5) {
      throwWithMessage("analyzeElectrons, test configuration error, expectedElectronIntegralValues must have size 5");
    }
    auto const& electrons = iEvent.get(electronsToken_);
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    if (electrons.size() != vectorSize) {
      throwWithMessage("analyzeElectrons, electrons does not have expected size");
    }
    unsigned int i = 0;
    for (auto const& electron : electrons) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      if (electron.pt() != expectedElectronFloatingPointValues_[0] + offset) {
        throwWithMessage("analyzeElectrons, pt does not equal expected value");
      }
      if (electron.eta() != expectedElectronFloatingPointValues_[1] + offset) {
        throwWithMessage("analyzeElectrons, eta does not equal expected value");
      }
      if (electron.phi() != expectedElectronFloatingPointValues_[2] + offset) {
        throwWithMessage("analyzeElectrons, phi does not equal expected value");
      }
      if (electron.m() != expectedElectronFloatingPointValues_[3] + offset) {
        throwWithMessage("analyzeElectrons, m does not equal expected value");
      }
      if (electron.d0() != expectedElectronFloatingPointValues_[4] + offset) {
        throwWithMessage("analyzeElectrons, d0 does not equal expected value");
      }
      if (electron.dz() != expectedElectronFloatingPointValues_[5] + offset) {
        throwWithMessage("analyzeElectrons,  dz does not equal expected value");
      }
      if (electron.dEtaIn() != expectedElectronFloatingPointValues_[6] + offset) {
        throwWithMessage("analyzeElectrons,  dEtaIn does not equal expected value");
      }
      if (electron.dPhiIn() != expectedElectronFloatingPointValues_[7] + offset) {
        throwWithMessage("analyzeElectrons, dPhiIn does not equal expected value");
      }
      if (electron.sigmaIetaIeta() != expectedElectronFloatingPointValues_[8] + offset) {
        throwWithMessage("analyzeElectrons, sigmaIetaIeta does not equal expected value");
      }
      if (electron.hOverE() != expectedElectronFloatingPointValues_[9] + offset) {
        throwWithMessage("analyzeElectrons, hOverE does not equal expected value");
      }
      if (electron.ooEMOop() != expectedElectronFloatingPointValues_[10] + offset) {
        throwWithMessage("analyzeElectrons, ooEMOop does not equal expected value");
      }
      if (electron.missingHits() != expectedElectronIntegralValues_[0] + iOffset) {
        throwWithMessage("analyzeElectrons, missingHits does not equal expected value");
      }
      if (electron.charge() != expectedElectronIntegralValues_[1] + iOffset) {
        throwWithMessage("analyzeElectrons, charge does not equal expected value");
      }
      if (electron.ecalIso() != expectedElectronFloatingPointValues_[11] + offset) {
        throwWithMessage("analyzeElectrons, ecalIso does not equal expected value");
      }
      if (electron.hcalIso() != expectedElectronFloatingPointValues_[12] + offset) {
        throwWithMessage("analyzeElectrons, hcalIso does not equal expected value");
      }
      if (electron.trackIso() != expectedElectronFloatingPointValues_[13] + offset) {
        throwWithMessage("analyzeElectrons, trackIso does not equal expected value");
      }
      if (electron.r9() != expectedElectronFloatingPointValues_[14] + offset) {
        throwWithMessage("analyzeElectrons, r9 does not equal expected value");
      }
      if (electron.sMin() != expectedElectronFloatingPointValues_[15] + offset) {
        throwWithMessage("analyzeElectrons, sMin does not equal expected value");
      }
      if (electron.sMaj() != expectedElectronFloatingPointValues_[16] + offset) {
        throwWithMessage("analyzeElectrons, sMaj does not equal expected value");
      }
      if (electron.seedId() != static_cast<unsigned int>(expectedElectronIntegralValues_[2] + iOffset)) {
        throwWithMessage("analyzeElectrons, seedId does not equal expected value");
      }
      if (electron.energyMatrix().size() != vectorSize) {
        throwWithMessage("analyzeElectrons, energyMatrix does not have expected size");
      }
      unsigned int j = 0;
      for (auto const& val : electron.energyMatrix()) {
        if (val != expectedElectronFloatingPointValues_[17] + offset + 10 * j) {
          throwWithMessage("analyzeElectrons, energyMatrix does not contain expected value");
        }
        ++j;
      }
      if (electron.detIds().size() != vectorSize) {
        throwWithMessage("analyzeElectrons, detIds does not have expected size");
      }
      j = 0;
      for (auto const& val : electron.detIds()) {
        if (val != expectedElectronIntegralValues_[3] + iOffset + 10 * j) {
          throwWithMessage("analyzeElectrons, detIds does not contain expected value");
        }
        ++j;
      }
      if (electron.timingMatrix().size() != vectorSize) {
        throwWithMessage("analyzeElectrons, timingMatrix does not have expected size");
      }
      j = 0;
      for (auto const& val : electron.timingMatrix()) {
        if (val != expectedElectronFloatingPointValues_[18] + offset + 10 * j) {
          throwWithMessage("analyzeElectrons, timingMatrix does not contain expected value");
        }
        ++j;
      }
      if (electron.rechitZeroSuppression() != static_cast<bool>((expectedElectronIntegralValues_[4] + iOffset) % 2)) {
        throwWithMessage("analyzeElectrons, rechitZeroSuppression does not equal expected value");
      }
      ++i;
    }
  }

  void TestReadRun3Scouting::analyzeMuons(edm::Event const& iEvent) const {
    if (expectedMuonFloatingPointValues_.size() != 37) {
      throwWithMessage("analyzeMuons, test configuration error, expectedMuonFloatingPointValues must have size 37");
    }
    if (expectedMuonIntegralValues_.size() != 26) {
      throwWithMessage("analyzeMuons, test configuration error, expectedMuonIntegralValues must have size 26");
    }
    auto const& muons = iEvent.get(muonsToken_);
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    if (muons.size() != vectorSize) {
      throwWithMessage("analyzeMuons, muons does not have expected size");
    }
    unsigned int i = 0;
    for (auto const& muon : muons) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      if (muon.pt() != expectedMuonFloatingPointValues_[0] + offset) {
        throwWithMessage("analyzeMuons, pt does not equal expected value");
      }
      if (muon.eta() != expectedMuonFloatingPointValues_[1] + offset) {
        throwWithMessage("analyzeMuons, eta does not equal expected value");
      }
      if (muon.phi() != expectedMuonFloatingPointValues_[2] + offset) {
        throwWithMessage("analyzeMuons, phi does not equal expected value");
      }
      if (muon.m() != expectedMuonFloatingPointValues_[3] + offset) {
        throwWithMessage("analyzeMuons, m does not equal expected value");
      }
      if (muon.type() != static_cast<unsigned int>(expectedMuonIntegralValues_[0] + iOffset)) {
        throwWithMessage("analyzeMuons, type does not equal expected value");
      }
      if (muon.charge() != expectedMuonIntegralValues_[1] + iOffset) {
        throwWithMessage("analyzeMuons, charge does not equal expected value");
      }
      if (muon.normalizedChi2() != expectedMuonFloatingPointValues_[4] + offset) {
        throwWithMessage("analyzeMuons,  normalizedChi2 does not equal expected value");
      }
      if (muon.ecalIso() != expectedMuonFloatingPointValues_[5] + offset) {
        throwWithMessage("analyzeMuons, ecalIso does not equal expected value");
      }
      if (muon.hcalIso() != expectedMuonFloatingPointValues_[6] + offset) {
        throwWithMessage("analyzeMuons, hcalIso does not equal expected value");
      }
      if (muon.trackIso() != expectedMuonFloatingPointValues_[7] + offset) {
        throwWithMessage("analyzeMuons, trackIso does not equal expected value");
      }
      if (muon.nValidStandAloneMuonHits() != expectedMuonIntegralValues_[2] + iOffset) {
        throwWithMessage("analyzeMuons, nValidStandAloneMuonHits does not equal expected value");
      }
      if (muon.nStandAloneMuonMatchedStations() != expectedMuonIntegralValues_[3] + iOffset) {
        throwWithMessage("analyzeMuons, nStandAloneMuonMatchedStations does not equal expected value");
      }
      if (muon.nValidRecoMuonHits() != expectedMuonIntegralValues_[4] + iOffset) {
        throwWithMessage("analyzeMuons, nValidRecoMuonHits does not equal expected value");
      }
      if (muon.nRecoMuonChambers() != expectedMuonIntegralValues_[5] + iOffset) {
        throwWithMessage("analyzeMuons, nRecoMuonChambers does not equal expected value");
      }
      if (muon.nRecoMuonChambersCSCorDT() != expectedMuonIntegralValues_[6] + iOffset) {
        throwWithMessage("analyzeMuons, nRecoMuonChambersCSCorDT does not equal expected value");
      }
      if (muon.nRecoMuonMatches() != expectedMuonIntegralValues_[7] + iOffset) {
        throwWithMessage("analyzeMuons, nRecoMuonMatches does not equal expected value");
      }
      if (muon.nRecoMuonMatchedStations() != expectedMuonIntegralValues_[8] + iOffset) {
        throwWithMessage("analyzeMuons, nRecoMuonMatchedStations does not equal expected value");
      }
      if (muon.nRecoMuonExpectedMatchedStations() !=
          static_cast<unsigned int>(expectedMuonIntegralValues_[9] + iOffset)) {
        throwWithMessage("analyzeMuons, nRecoMuonExpectedMatchedStations does not equal expected value");
      }
      if (muon.recoMuonStationMask() != static_cast<unsigned int>(expectedMuonIntegralValues_[10] + iOffset)) {
        throwWithMessage("analyzeMuons, recoMuonStationMask does not equal expected value");
      }
      if (muon.nRecoMuonMatchedRPCLayers() != expectedMuonIntegralValues_[11] + iOffset) {
        throwWithMessage("analyzeMuons, nRecoMuonMatchedRPCLayers does not equal expected value");
      }
      if (muon.recoMuonRPClayerMask() != static_cast<unsigned int>(expectedMuonIntegralValues_[12] + iOffset)) {
        throwWithMessage("analyzeMuons, recoMuonRPClayerMask does not equal expected value");
      }
      if (muon.nValidPixelHits() != expectedMuonIntegralValues_[13] + iOffset) {
        throwWithMessage("analyzeMuons, nValidPixelHits does not equal expected value");
      }
      if (muon.nValidStripHits() != expectedMuonIntegralValues_[14] + iOffset) {
        throwWithMessage("analyzeMuons, nValidStripHits does not equal expected value");
      }
      if (muon.nPixelLayersWithMeasurement() != expectedMuonIntegralValues_[15] + iOffset) {
        throwWithMessage("analyzeMuons, nPixelLayersWithMeasurement does not equal expected value");
      }
      if (muon.nTrackerLayersWithMeasurement() != expectedMuonIntegralValues_[16] + iOffset) {
        throwWithMessage("analyzeMuons, nTrackerLayersWithMeasurement does not equal expected value");
      }
      if (muon.trk_chi2() != expectedMuonFloatingPointValues_[8] + offset) {
        throwWithMessage("analyzeMuons, trk_chi2  does not equal expected value");
      }
      if (muon.trk_ndof() != expectedMuonFloatingPointValues_[9] + offset) {
        throwWithMessage("analyzeMuons, trk_ndof does not equal expected value");
      }
      if (muon.trk_dxy() != expectedMuonFloatingPointValues_[10] + offset) {
        throwWithMessage("analyzeMuons, trk_dxy does not equal expected value");
      }
      if (muon.trk_dz() != expectedMuonFloatingPointValues_[11] + offset) {
        throwWithMessage("analyzeMuons, trk_dz does not equal expected value");
      }
      if (muon.trk_qoverp() != expectedMuonFloatingPointValues_[12] + offset) {
        throwWithMessage("analyzeMuons, trk_qoverp does not equal expected value");
      }
      if (muon.trk_lambda() != expectedMuonFloatingPointValues_[13] + offset) {
        throwWithMessage("analyzeMuons, trk_lambda does not equal expected value");
      }
      if (muon.trk_pt() != expectedMuonFloatingPointValues_[14] + offset) {
        throwWithMessage("analyzeMuons, trk_pt does not equal expected value");
      }
      if (muon.trk_phi() != expectedMuonFloatingPointValues_[15] + offset) {
        throwWithMessage("analyzeMuons, trk_phi does not equal expected value");
      }
      if (muon.trk_eta() != expectedMuonFloatingPointValues_[16] + offset) {
        throwWithMessage("analyzeMuons, trk_eta does not equal expected value");
      }
      if (muon.trk_dxyError() != expectedMuonFloatingPointValues_[17] + offset) {
        throwWithMessage("analyzeMuons, trk_dxyError does not equal expected value");
      }
      if (muon.trk_dzError() != expectedMuonFloatingPointValues_[18] + offset) {
        throwWithMessage("analyzeMuons, trk_dzError does not equal expected value");
      }
      if (muon.trk_qoverpError() != expectedMuonFloatingPointValues_[19] + offset) {
        throwWithMessage("analyzeMuons, trk_qoverpError does not equal expected value");
      }
      if (muon.trk_lambdaError() != expectedMuonFloatingPointValues_[20] + offset) {
        throwWithMessage("analyzeMuons, trk_lambdaError does not equal expected value");
      }
      if (muon.trk_phiError() != expectedMuonFloatingPointValues_[21] + offset) {
        throwWithMessage("analyzeMuons, trk_phiError does not equal expected value");
      }
      if (muon.trk_dsz() != expectedMuonFloatingPointValues_[22] + offset) {
        throwWithMessage("analyzeMuons, trk_dsz does not equal expected value");
      }
      if (muon.trk_dszError() != expectedMuonFloatingPointValues_[23] + offset) {
        throwWithMessage("analyzeMuons, trk_dszError does not equal expected value");
      }
      if (muon.trk_qoverp_lambda_cov() != expectedMuonFloatingPointValues_[24] + offset) {
        throwWithMessage("analyzeMuons, trk_qoverp_lambda_cov does not equal expected value");
      }
      if (muon.trk_qoverp_phi_cov() != expectedMuonFloatingPointValues_[25] + offset) {
        throwWithMessage("analyzeMuons, trk_qoverp_phi_cov does not equal expected value");
      }
      if (muon.trk_qoverp_dxy_cov() != expectedMuonFloatingPointValues_[26] + offset) {
        throwWithMessage("analyzeMuons, trk_qoverp_dxy_cov does not equal expected value");
      }
      if (muon.trk_qoverp_dsz_cov() != expectedMuonFloatingPointValues_[27] + offset) {
        throwWithMessage("analyzeMuons, trk_qoverp_dsz_cov does not equal expected value");
      }
      if (muon.trk_lambda_phi_cov() != expectedMuonFloatingPointValues_[28] + offset) {
        throwWithMessage("analyzeMuons, trk_lambda_phi_cov does not equal expected value");
      }
      if (muon.trk_lambda_dxy_cov() != expectedMuonFloatingPointValues_[29] + offset) {
        throwWithMessage("analyzeMuons, trk_lambda_dxy_cov  does not equal expected value");
      }
      if (muon.trk_lambda_dsz_cov() != expectedMuonFloatingPointValues_[30] + offset) {
        throwWithMessage("analyzeMuons, trk_lambda_dsz_cov  does not equal expected value");
      }
      if (muon.trk_phi_dxy_cov() != expectedMuonFloatingPointValues_[31] + offset) {
        throwWithMessage("analyzeMuons, trk_phi_dxy_cov does not equal expected value");
      }
      if (muon.trk_phi_dsz_cov() != expectedMuonFloatingPointValues_[32] + offset) {
        throwWithMessage("analyzeMuons, trk_phi_dsz_cov does not equal expected value");
      }
      if (muon.trk_dxy_dsz_cov() != expectedMuonFloatingPointValues_[33] + offset) {
        throwWithMessage("analyzeMuons, trk_dxy_dsz_cov does not equal expected value");
      }
      if (muon.trk_vx() != expectedMuonFloatingPointValues_[34] + offset) {
        throwWithMessage("analyzeMuons, trk_vx does not equal expected value");
      }
      if (muon.trk_vy() != expectedMuonFloatingPointValues_[35] + offset) {
        throwWithMessage("analyzeMuons, trk_vy does not equal expected value");
      }
      if (muon.trk_vz() != expectedMuonFloatingPointValues_[36] + offset) {
        throwWithMessage("analyzeMuons, trk_vz does not equal expected value");
      }
      int j = 0;
      for (auto const& val : muon.vtxIndx()) {
        if (val != expectedMuonIntegralValues_[17] + iOffset + 10 * j) {
          throwWithMessage("analyzeMuons, vtxIndx does not contain expected value");
        }
        ++j;
      }
      if (muon.trk_hitPattern().hitCount != static_cast<uint8_t>(expectedMuonIntegralValues_[18] + iOffset)) {
        throwWithMessage("analyzeMuons, hitCount does not equal expected value");
      }
      if (muon.trk_hitPattern().beginTrackHits != static_cast<uint8_t>(expectedMuonIntegralValues_[19] + iOffset)) {
        throwWithMessage("analyzeMuons, beginTrackHits does not equal expected value");
      }
      if (muon.trk_hitPattern().endTrackHits != static_cast<uint8_t>(expectedMuonIntegralValues_[20] + iOffset)) {
        throwWithMessage("analyzeMuons, endTrackHits does not equal expected value");
      }
      if (muon.trk_hitPattern().beginInner != static_cast<uint8_t>(expectedMuonIntegralValues_[21] + iOffset)) {
        throwWithMessage("analyzeMuons, beginInner does not equal expected value");
      }
      if (muon.trk_hitPattern().endInner != static_cast<uint8_t>(expectedMuonIntegralValues_[22] + iOffset)) {
        throwWithMessage("analyzeMuons, endInner does not equal expected value");
      }
      if (muon.trk_hitPattern().beginOuter != static_cast<uint8_t>(expectedMuonIntegralValues_[23] + iOffset)) {
        throwWithMessage("analyzeMuons, beginOuter does not equal expected value");
      }
      if (muon.trk_hitPattern().endOuter != static_cast<uint8_t>(expectedMuonIntegralValues_[24] + iOffset)) {
        throwWithMessage("analyzeMuons, endOuter does not equal expected value");
      }
      j = 0;
      for (auto const& val : muon.trk_hitPattern().hitPattern) {
        if (val != static_cast<uint16_t>(expectedMuonIntegralValues_[25] + iOffset + 10 * j)) {
          throwWithMessage("analyzeMuons, hitPattern does not contain expected value");
        }
        ++j;
      }
      ++i;
    }
  }

  void TestReadRun3Scouting::analyzeParticles(edm::Event const& iEvent) const {
    if (expectedParticleFloatingPointValues_.size() != 11) {
      throwWithMessage(
          "analyzeParticles, test configuration error, expectedParticleFloatingPointValues must have size 11");
    }
    if (expectedParticleIntegralValues_.size() != 5) {
      throwWithMessage("analyzeParticles, test configuration error, expectedParticleIntegralValues must have size 5");
    }
    auto const& particles = iEvent.get(particlesToken_);
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    if (particles.size() != vectorSize) {
      throwWithMessage("analyzeParticles, particles does not have expected size");
    }
    unsigned int i = 0;
    for (auto const& particle : particles) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      if (particle.pt() != expectedParticleFloatingPointValues_[0] + offset) {
        throwWithMessage("analyzeParticles, pt does not equal expected value");
      }
      if (particle.eta() != expectedParticleFloatingPointValues_[1] + offset) {
        throwWithMessage("analyzeParticles, eta does not equal expected value");
      }
      if (particle.phi() != expectedParticleFloatingPointValues_[2] + offset) {
        throwWithMessage("analyzeParticles, phi does not equal expected value");
      }
      if (particle.pdgId() != expectedParticleIntegralValues_[0] + iOffset) {
        throwWithMessage("analyzeParticles, pdgId does not equal expected value");
      }
      if (particle.vertex() != expectedParticleIntegralValues_[1] + iOffset) {
        throwWithMessage("analyzeParticles, vertex does not equal expected value");
      }
      if (particle.normchi2() != expectedParticleFloatingPointValues_[3] + offset) {
        throwWithMessage("analyzeParticles, normchi2 does not equal expected value");
      }
      if (particle.dz() != expectedParticleFloatingPointValues_[4] + offset) {
        throwWithMessage("analyzeParticles, dz does not equal expected value");
      }
      if (particle.dxy() != expectedParticleFloatingPointValues_[5] + offset) {
        throwWithMessage("analyzeParticles, dxy does not equal expected value");
      }
      if (particle.dzsig() != expectedParticleFloatingPointValues_[6] + offset) {
        throwWithMessage("analyzeParticles, dzsig does not equal expected value");
      }
      if (particle.dxysig() != expectedParticleFloatingPointValues_[7] + offset) {
        throwWithMessage("analyzeParticles, dxysig does not equal expected value");
      }
      if (particle.lostInnerHits() != static_cast<uint8_t>(expectedParticleIntegralValues_[2] + iOffset)) {
        throwWithMessage("analyzeParticles, lostInnerHits does not equal expected value");
      }
      if (particle.quality() != static_cast<uint8_t>(expectedParticleIntegralValues_[3] + iOffset)) {
        throwWithMessage("analyzeParticles, quality does not equal expected value");
      }
      if (particle.trk_pt() != expectedParticleFloatingPointValues_[8] + offset) {
        throwWithMessage("analyzeParticles, trk_pt does not equal expected value");
      }
      if (particle.trk_eta() != expectedParticleFloatingPointValues_[9] + offset) {
        throwWithMessage("analyzeParticles, trk_eta does not equal expected value");
      }
      if (particle.trk_phi() != expectedParticleFloatingPointValues_[10] + offset) {
        throwWithMessage("analyzeParticles, trk_phi does not equal expected value");
      }
      if (particle.relative_trk_vars() != static_cast<bool>((expectedParticleIntegralValues_[4] + iOffset) % 2)) {
        throwWithMessage("analyzeParticles, relative_trk_vars does not equal expected value");
      }
      ++i;
    }
  }

  void TestReadRun3Scouting::analyzePFJets(edm::Event const& iEvent) const {
    if (expectedPFJetFloatingPointValues_.size() != 15) {
      throwWithMessage("analyzePFJets, test configuration error, expectedPFJetFloatingPointValues must have size 15");
    }
    if (expectedPFJetIntegralValues_.size() != 8) {
      throwWithMessage("analyzePFJets, test configuration error, expectedPFJetIntegralValues must have size 8");
    }
    auto const& pfJets = iEvent.get(pfJetsToken_);
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    if (pfJets.size() != vectorSize) {
      throwWithMessage("analyzePFJets, pfJets does not have expected size");
    }
    unsigned int i = 0;
    for (auto const& pfJet : pfJets) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      if (pfJet.pt() != expectedPFJetFloatingPointValues_[0] + offset) {
        throwWithMessage("analyzePFJets, pt does not equal expected value");
      }
      if (pfJet.eta() != expectedPFJetFloatingPointValues_[1] + offset) {
        throwWithMessage("analyzePFJets, eta does not equal expected value");
      }
      if (pfJet.phi() != expectedPFJetFloatingPointValues_[2] + offset) {
        throwWithMessage("analyzePFJets, phi does not equal expected value");
      }
      if (pfJet.m() != expectedPFJetFloatingPointValues_[3] + offset) {
        throwWithMessage("analyzePFJets, m does not equal expected value");
      }
      if (pfJet.jetArea() != expectedPFJetFloatingPointValues_[4] + offset) {
        throwWithMessage("analyzePFJets, jetArea does not equal expected value");
      }
      if (pfJet.chargedHadronEnergy() != expectedPFJetFloatingPointValues_[5] + offset) {
        throwWithMessage("analyzePFJets, chargedHadronEnergy does not equal expected value");
      }
      if (pfJet.neutralHadronEnergy() != expectedPFJetFloatingPointValues_[6] + offset) {
        throwWithMessage("analyzePFJets, neutralHadronEnergy does not equal expected value");
      }
      if (pfJet.photonEnergy() != expectedPFJetFloatingPointValues_[7] + offset) {
        throwWithMessage("analyzePFJets, photonEnergy does not equal expected value");
      }
      if (pfJet.electronEnergy() != expectedPFJetFloatingPointValues_[8] + offset) {
        throwWithMessage("analyzePFJets, electronEnergy does not equal expected value");
      }
      if (pfJet.muonEnergy() != expectedPFJetFloatingPointValues_[9] + offset) {
        throwWithMessage("analyzePFJets, muonEnergy does not equal expected value");
      }
      if (pfJet.HFHadronEnergy() != expectedPFJetFloatingPointValues_[10] + offset) {
        throwWithMessage("analyzePFJets, HFHadronEnergy does not equal expected value");
      }
      if (pfJet.HFEMEnergy() != expectedPFJetFloatingPointValues_[11] + offset) {
        throwWithMessage("analyzePFJets, HFEMEnergy does not equal expected value");
      }
      if (pfJet.chargedHadronMultiplicity() != expectedPFJetIntegralValues_[0] + iOffset) {
        throwWithMessage("analyzePFJets, chargedHadronMultiplicity does not equal expected value");
      }
      if (pfJet.neutralHadronMultiplicity() != expectedPFJetIntegralValues_[1] + iOffset) {
        throwWithMessage("analyzePFJets, neutralHadronMultiplicity does not equal expected value");
      }
      if (pfJet.photonMultiplicity() != expectedPFJetIntegralValues_[2] + iOffset) {
        throwWithMessage("analyzePFJets, photonMultiplicity does not equal expected value");
      }
      if (pfJet.electronMultiplicity() != expectedPFJetIntegralValues_[3] + iOffset) {
        throwWithMessage("analyzePFJets, electronMultiplicity does not equal expected value");
      }
      if (pfJet.muonMultiplicity() != expectedPFJetIntegralValues_[4] + iOffset) {
        throwWithMessage("analyzePFJets, muonMultiplicity does not equal expected value");
      }
      if (pfJet.HFHadronMultiplicity() != expectedPFJetIntegralValues_[5] + iOffset) {
        throwWithMessage("analyzePFJets, HFHadronMultiplicity does not equal expected value");
      }
      if (pfJet.HFEMMultiplicity() != expectedPFJetIntegralValues_[6] + iOffset) {
        throwWithMessage("analyzePFJets, HFEMMultiplicity does not equal expected value");
      }
      if (pfJet.HOEnergy() != expectedPFJetFloatingPointValues_[12] + offset) {
        throwWithMessage("analyzePFJets, HOEnergy does not equal expected value");
      }
      if (pfJet.csv() != expectedPFJetFloatingPointValues_[13] + offset) {
        throwWithMessage("analyzePFJets, csv does not equal expected value");
      }
      if (pfJet.mvaDiscriminator() != expectedPFJetFloatingPointValues_[14] + offset) {
        throwWithMessage("analyzePFJets, mvaDiscriminator does not equal expected value");
      }
      int j = 0;
      for (auto const& val : pfJet.constituents()) {
        if (val != expectedPFJetIntegralValues_[7] + iOffset + 10 * j) {
          throwWithMessage("analyzePFJets, constituents does not contain expected value");
        }
        ++j;
      }
      ++i;
    }
  }

  void TestReadRun3Scouting::analyzePhotons(edm::Event const& iEvent) const {
    if (expectedPhotonFloatingPointValues_.size() != 14) {
      throwWithMessage("analyzePhotons, test configuration error, expectedPhotonFloatingPointValues must have size 14");
    }
    if (expectedPhotonIntegralValues_.size() != 3) {
      throwWithMessage("analyzePhotons, test configuration error, expectedPhotonIntegralValues must have size 3");
    }
    auto const& photons = iEvent.get(photonsToken_);
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    if (photons.size() != vectorSize) {
      throwWithMessage("analyzePhotons, photons does not have expected size");
    }
    unsigned int i = 0;
    for (auto const& photon : photons) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      if (photon.pt() != expectedPhotonFloatingPointValues_[0] + offset) {
        throwWithMessage("analyzePhotons, pt does not equal expected value");
      }
      if (photon.eta() != expectedPhotonFloatingPointValues_[1] + offset) {
        throwWithMessage("analyzePhotons, eta does not equal expected value");
      }
      if (photon.phi() != expectedPhotonFloatingPointValues_[2] + offset) {
        throwWithMessage("analyzePhotons, phi does not equal expected value");
      }
      if (photon.m() != expectedPhotonFloatingPointValues_[3] + offset) {
        throwWithMessage("analyzePhotons, m does not equal expected value");
      }
      if (photon.sigmaIetaIeta() != expectedPhotonFloatingPointValues_[4] + offset) {
        throwWithMessage("analyzePhotons, sigmaIetaIeta does not equal expected value");
      }
      if (photon.hOverE() != expectedPhotonFloatingPointValues_[5] + offset) {
        throwWithMessage("analyzePhotons, hOverE does not equal expected value");
      }
      if (photon.ecalIso() != expectedPhotonFloatingPointValues_[6] + offset) {
        throwWithMessage("analyzePhotons, ecalIso does not equal expected value");
      }
      if (photon.hcalIso() != expectedPhotonFloatingPointValues_[7] + offset) {
        throwWithMessage("analyzePhotons, hcalIso does not equal expected value");
      }
      if (photon.trkIso() != expectedPhotonFloatingPointValues_[8] + offset) {
        throwWithMessage("analyzePhotons, trkIso does not equal expected value");
      }
      if (photon.r9() != expectedPhotonFloatingPointValues_[9] + offset) {
        throwWithMessage("analyzePhotons, r9 does not equal expected value");
      }
      if (photon.sMin() != expectedPhotonFloatingPointValues_[10] + offset) {
        throwWithMessage("analyzePhotons, sMin does not equal expected value");
      }
      if (photon.sMaj() != expectedPhotonFloatingPointValues_[11] + offset) {
        throwWithMessage("analyzePhotons, sMaj does not equal expected value");
      }
      if (photon.seedId() != static_cast<unsigned int>(expectedPhotonIntegralValues_[0] + iOffset)) {
        throwWithMessage("analyzePhotons, seedId does not equal expected value");
      }

      if (photon.energyMatrix().size() != vectorSize) {
        throwWithMessage("analyzePhotons, energyMatrix does not have expected size");
      }
      unsigned int j = 0;
      for (auto const& val : photon.energyMatrix()) {
        if (val != expectedPhotonFloatingPointValues_[12] + offset + 10 * j) {
          throwWithMessage("analyzePhotons, energyMatrix does not contain expected value");
        }
        ++j;
      }
      if (photon.detIds().size() != vectorSize) {
        throwWithMessage("analyzePhotons, detIds does not have expected size");
      }
      j = 0;
      for (auto const& val : photon.detIds()) {
        if (val != static_cast<uint32_t>(expectedPhotonIntegralValues_[1] + iOffset + 10 * j)) {
          throwWithMessage("analyzePhotons, detIds does not contain expected value");
        }
        ++j;
      }
      if (photon.timingMatrix().size() != vectorSize) {
        throwWithMessage("analyzePhotons, timingMatrix does not have expected size");
      }
      j = 0;
      for (auto const& val : photon.timingMatrix()) {
        if (val != expectedPhotonFloatingPointValues_[13] + offset + 10 * j) {
          throwWithMessage("analyzePhotons, timingMatrix does not contain expected value");
        }
        ++j;
      }
      if (photon.rechitZeroSuppression() != static_cast<bool>((expectedPhotonIntegralValues_[2] + iOffset) % 2)) {
        throwWithMessage("analyzePhotons, rechitZeroSuppression does not equal expected value");
      }
      ++i;
    }
  }

  void TestReadRun3Scouting::analyzeTracks(edm::Event const& iEvent) const {
    if (expectedTrackFloatingPointValues_.size() != 29) {
      throwWithMessage("analyzeTracks, test configuration error, expectedTrackFloatingPointValues must have size 29");
    }
    if (expectedTrackIntegralValues_.size() != 5) {
      throwWithMessage("analyzeTracks, test configuration error, expectedTrackIntegralValues must have size 5");
    }
    auto const& tracks = iEvent.get(tracksToken_);
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    if (tracks.size() != vectorSize) {
      throwWithMessage("analyzeTracks, tracks does not have expected size");
    }
    unsigned int i = 0;
    for (auto const& track : tracks) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      if (track.tk_pt() != expectedTrackFloatingPointValues_[0] + offset) {
        throwWithMessage("analyzeTracks, tk_pt does not equal expected value");
      }
      if (track.tk_eta() != expectedTrackFloatingPointValues_[1] + offset) {
        throwWithMessage("analyzeTracks, tk_eta does not equal expected value");
      }
      if (track.tk_phi() != expectedTrackFloatingPointValues_[2] + offset) {
        throwWithMessage("analyzeTracks, tk_phi does not equal expected value");
      }
      if (track.tk_chi2() != expectedTrackFloatingPointValues_[3] + offset) {
        throwWithMessage("analyzeTracks, tk_chi2 does not equal expected value");
      }
      if (track.tk_ndof() != expectedTrackFloatingPointValues_[4] + offset) {
        throwWithMessage("analyzeTracks, tk_ndof does not equal expected value");
      }
      if (track.tk_charge() != expectedTrackIntegralValues_[0] + iOffset) {
        throwWithMessage("analyzeTracks, tk_charge does not equal expected value");
      }
      if (track.tk_dxy() != expectedTrackFloatingPointValues_[5] + offset) {
        throwWithMessage("analyzeTracks, tk_dxy does not equal expected value");
      }
      if (track.tk_dz() != expectedTrackFloatingPointValues_[6] + offset) {
        throwWithMessage("analyzeTracks, tk_dz does not equal expected value");
      }
      if (track.tk_nValidPixelHits() != expectedTrackIntegralValues_[1] + iOffset) {
        throwWithMessage("analyzeTracks, tk_nValidPixelHits does not equal expected value");
      }
      if (track.tk_nTrackerLayersWithMeasurement() != expectedTrackIntegralValues_[2] + iOffset) {
        throwWithMessage("analyzeTracks, tk_nTrackerLayersWithMeasurement does not equal expected value");
      }
      if (track.tk_nValidStripHits() != expectedTrackIntegralValues_[3] + iOffset) {
        throwWithMessage("analyzeTracks, tk_nValidStripHits does not equal expected value");
      }
      if (track.tk_qoverp() != expectedTrackFloatingPointValues_[7] + offset) {
        throwWithMessage("analyzeTracks, tk_qoverp does not equal expected value");
      }
      if (track.tk_lambda() != expectedTrackFloatingPointValues_[8] + offset) {
        throwWithMessage("analyzeTracks, tk_lambda does not equal expected value");
      }
      if (track.tk_dxy_Error() != expectedTrackFloatingPointValues_[9] + offset) {
        throwWithMessage("analyzeTracks, tk_dxy_Error does not equal expected value");
      }
      if (track.tk_dz_Error() != expectedTrackFloatingPointValues_[10] + offset) {
        throwWithMessage("analyzeTracks, tk_dz_Error does not equal expected value");
      }
      if (track.tk_qoverp_Error() != expectedTrackFloatingPointValues_[11] + offset) {
        throwWithMessage("analyzeTracks, tk_qoverp_Error does not equal expected value");
      }
      if (track.tk_lambda_Error() != expectedTrackFloatingPointValues_[12] + offset) {
        throwWithMessage("analyzeTracks, tk_lambda_Error does not equal expected value");
      }
      if (track.tk_phi_Error() != expectedTrackFloatingPointValues_[13] + offset) {
        throwWithMessage("analyzeTracks, tk_phi_Error does not equal expected value");
      }
      if (track.tk_dsz() != expectedTrackFloatingPointValues_[14] + offset) {
        throwWithMessage("analyzeTracks, tk_dsz does not equal expected value");
      }
      if (track.tk_dsz_Error() != expectedTrackFloatingPointValues_[15] + offset) {
        throwWithMessage("analyzeTracks, tk_dsz_Error does not equal expected value");
      }
      if (track.tk_qoverp_lambda_cov() != expectedTrackFloatingPointValues_[16] + offset) {
        throwWithMessage("analyzeTracks, tk_qoverp_lambda_cov does not equal expected value");
      }
      if (track.tk_qoverp_phi_cov() != expectedTrackFloatingPointValues_[17] + offset) {
        throwWithMessage("analyzeTracks, tk_qoverp_phi_cov does not equal expected value");
      }
      if (track.tk_qoverp_dxy_cov() != expectedTrackFloatingPointValues_[18] + offset) {
        throwWithMessage("analyzeTracks, tk_qoverp_dxy_cov does not equal expected value");
      }
      if (track.tk_qoverp_dsz_cov() != expectedTrackFloatingPointValues_[19] + offset) {
        throwWithMessage("analyzeTracks, tk_qoverp_dsz_cov does not equal expected value");
      }
      if (track.tk_lambda_phi_cov() != expectedTrackFloatingPointValues_[20] + offset) {
        throwWithMessage("analyzeTracks, tk_lambda_phi_cov does not equal expected value");
      }
      if (track.tk_lambda_dxy_cov() != expectedTrackFloatingPointValues_[21] + offset) {
        throwWithMessage("analyzeTracks, tk_lambda_dxy_cov does not equal expected value");
      }
      if (track.tk_lambda_dsz_cov() != expectedTrackFloatingPointValues_[22] + offset) {
        throwWithMessage("analyzeTracks, tk_lambda_dsz_cov does not equal expected value");
      }
      if (track.tk_phi_dxy_cov() != expectedTrackFloatingPointValues_[23] + offset) {
        throwWithMessage("analyzeTracks, tk_phi_dxy_cov does not equal expected value");
      }
      if (track.tk_phi_dsz_cov() != expectedTrackFloatingPointValues_[24] + offset) {
        throwWithMessage("analyzeTracks, tk_phi_dsz_cov does not equal expected value");
      }
      if (track.tk_dxy_dsz_cov() != expectedTrackFloatingPointValues_[25] + offset) {
        throwWithMessage("analyzeTracks, tk_dxy_dsz_cov does not equal expected value");
      }
      if (track.tk_vtxInd() != expectedTrackIntegralValues_[4] + iOffset) {
        throwWithMessage("analyzeTracks, tk_vtxInd does not equal expected value");
      }
      if (track.tk_vx() != expectedTrackFloatingPointValues_[26] + offset) {
        throwWithMessage("analyzeTracks, tk_vx does not equal expected value");
      }
      if (track.tk_vy() != expectedTrackFloatingPointValues_[27] + offset) {
        throwWithMessage("analyzeTracks, tk_vy does not equal expected value");
      }
      if (track.tk_vz() != expectedTrackFloatingPointValues_[28] + offset) {
        throwWithMessage("analyzeTracks, float tk_vz does not equal expected value");
      }
      ++i;
    }
  }

  void TestReadRun3Scouting::analyzeVertexes(edm::Event const& iEvent) const {
    if (expectedVertexFloatingPointValues_.size() != 7) {
      throwWithMessage("analyzeVertexes, test configuration error, expectedVertexFloatingPointValues must have size 7");
    }
    if (expectedVertexIntegralValues_.size() != 3) {
      throwWithMessage("analyzeVertexes, test configuration error, expectedVertexIntegralValues must have size 3");
    }
    auto const& vertexes = iEvent.get(vertexesToken_);
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    if (vertexes.size() != vectorSize) {
      throwWithMessage("analyzeVertexes, vertexes does not have expected size");
    }
    unsigned int i = 0;
    for (auto const& vertex : vertexes) {
      double offset = static_cast<double>(iEvent.id().event() + i);
      int iOffset = static_cast<int>(iEvent.id().event() + i);

      if (vertex.x() != expectedVertexFloatingPointValues_[0] + offset) {
        throwWithMessage("analyzeVertexes, x does not equal expected value");
      }
      if (vertex.y() != expectedVertexFloatingPointValues_[1] + offset) {
        throwWithMessage("analyzeVertexes, y does not equal expected value");
      }
      if (vertex.z() != expectedVertexFloatingPointValues_[2] + offset) {
        throwWithMessage("analyzeVertexes, z does not equal expected value");
      }
      if (vertex.zError() != expectedVertexFloatingPointValues_[3] + offset) {
        throwWithMessage("analyzeVertexes, zError does not equal expected value");
      }
      if (vertex.xError() != expectedVertexFloatingPointValues_[4] + offset) {
        throwWithMessage("analyzeVertexes, xError does not equal expected value");
      }
      if (vertex.yError() != expectedVertexFloatingPointValues_[5] + offset) {
        throwWithMessage("analyzeVertexes, yError does not equal expected value");
      }
      if (vertex.tracksSize() != expectedVertexIntegralValues_[0] + iOffset) {
        throwWithMessage("analyzeVertexes, tracksSize does not equal expected value");
      }
      if (vertex.chi2() != expectedVertexFloatingPointValues_[6] + offset) {
        throwWithMessage("analyzeVertexes, chi2 does not equal expected value");
      }
      if (vertex.ndof() != expectedVertexIntegralValues_[1] + iOffset) {
        throwWithMessage("analyzeVertexes, ndof does not equal expected value");
      }
      if (vertex.isValidVtx() != static_cast<bool>((expectedVertexIntegralValues_[2] + iOffset) % 2)) {
        throwWithMessage("analyzeVertexes, isValidVtx does not equal expected value");
      }
      ++i;
    }
  }

  void TestReadRun3Scouting::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadRun3Scouting::analyze, " << msg;
  }

}  // namespace edmtest

using edmtest::TestReadRun3Scouting;
DEFINE_FWK_MODULE(TestReadRun3Scouting);
