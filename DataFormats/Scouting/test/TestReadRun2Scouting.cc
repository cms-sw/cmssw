// -*- C++ -*-
//
// Package:    DataFormats/Scouting
// Class:      TestReadRun2Scouting
//
/**\class edmtest::TestReadRun2Scouting
  Description: Used as part of tests that ensure the run 2 Scouting
  data formats can be persistently written and in a subsequent process
  read. First, this is done using the current release version for writing
  and reading. In addition, the output file of the write process should
  be saved permanently each time a run 2 Scouting persistent data
  format changes. In unit tests, we read each of those saved files to verify
  that the current releases can read older versions of these data formats.
*/
// Original Author:  W. David Dagenhart
//         Created:  2 June 2023

#include "DataFormats/Scouting/interface/ScoutingCaloJet.h"
#include "DataFormats/Scouting/interface/ScoutingElectron.h"
#include "DataFormats/Scouting/interface/ScoutingMuon.h"
#include "DataFormats/Scouting/interface/ScoutingParticle.h"
#include "DataFormats/Scouting/interface/ScoutingPFJet.h"
#include "DataFormats/Scouting/interface/ScoutingPhoton.h"
#include "DataFormats/Scouting/interface/ScoutingTrack.h"
#include "DataFormats/Scouting/interface/ScoutingVertex.h"
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

  class TestReadRun2Scouting : public edm::global::EDAnalyzer<> {
  public:
    TestReadRun2Scouting(edm::ParameterSet const&);
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

    void throwWithMessageFromConstructor(const char*) const;
    void throwWithMessage(const char*) const;

    // These expected values are meaningless other than we use them
    // to check that values read from persistent storage match the values
    // we know were written.

    const std::vector<double> expectedCaloJetsValues_;
    const edm::EDGetTokenT<std::vector<ScoutingCaloJet>> caloJetsToken_;

    const std::vector<double> expectedElectronFloatingPointValues_;
    const std::vector<int> expectedElectronIntegralValues_;
    const edm::EDGetTokenT<std::vector<ScoutingElectron>> electronsToken_;

    const int inputMuonClassVersion_;
    const std::vector<double> expectedMuonFloatingPointValues_;
    const std::vector<int> expectedMuonIntegralValues_;
    const edm::EDGetTokenT<std::vector<ScoutingMuon>> muonsToken_;

    const std::vector<double> expectedParticleFloatingPointValues_;
    const std::vector<int> expectedParticleIntegralValues_;
    const edm::EDGetTokenT<std::vector<ScoutingParticle>> particlesToken_;

    const std::vector<double> expectedPFJetFloatingPointValues_;
    const std::vector<int> expectedPFJetIntegralValues_;
    const edm::EDGetTokenT<std::vector<ScoutingPFJet>> pfJetsToken_;

    const std::vector<double> expectedPhotonFloatingPointValues_;
    const edm::EDGetTokenT<std::vector<ScoutingPhoton>> photonsToken_;

    const int inputTrackClassVersion_;
    const std::vector<double> expectedTrackFloatingPointValues_;
    const std::vector<int> expectedTrackIntegralValues_;
    const edm::EDGetTokenT<std::vector<ScoutingTrack>> tracksToken_;

    const int inputVertexClassVersion_;
    const std::vector<double> expectedVertexFloatingPointValues_;
    const std::vector<int> expectedVertexIntegralValues_;
    const edm::EDGetTokenT<std::vector<ScoutingVertex>> vertexesToken_;
  };

  TestReadRun2Scouting::TestReadRun2Scouting(edm::ParameterSet const& iPSet)
      : expectedCaloJetsValues_(iPSet.getParameter<std::vector<double>>("expectedCaloJetsValues")),
        caloJetsToken_(consumes(iPSet.getParameter<edm::InputTag>("caloJetsTag"))),
        expectedElectronFloatingPointValues_(
            iPSet.getParameter<std::vector<double>>("expectedElectronFloatingPointValues")),
        expectedElectronIntegralValues_(iPSet.getParameter<std::vector<int>>("expectedElectronIntegralValues")),
        electronsToken_(consumes(iPSet.getParameter<edm::InputTag>("electronsTag"))),
        inputMuonClassVersion_(iPSet.getParameter<int>("muonClassVersion")),
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
        photonsToken_(consumes(iPSet.getParameter<edm::InputTag>("photonsTag"))),
        inputTrackClassVersion_(iPSet.getParameter<int>("trackClassVersion")),
        expectedTrackFloatingPointValues_(iPSet.getParameter<std::vector<double>>("expectedTrackFloatingPointValues")),
        expectedTrackIntegralValues_(iPSet.getParameter<std::vector<int>>("expectedTrackIntegralValues")),
        tracksToken_(consumes(iPSet.getParameter<edm::InputTag>("tracksTag"))),
        inputVertexClassVersion_(iPSet.getParameter<int>("vertexClassVersion")),
        expectedVertexFloatingPointValues_(
            iPSet.getParameter<std::vector<double>>("expectedVertexFloatingPointValues")),
        expectedVertexIntegralValues_(iPSet.getParameter<std::vector<int>>("expectedVertexIntegralValues")),
        vertexesToken_(consumes(iPSet.getParameter<edm::InputTag>("vertexesTag"))) {
    if (expectedCaloJetsValues_.size() != 16) {
      throwWithMessageFromConstructor("test configuration error, expectedCaloJetsValues must have size 16");
    }
    if (expectedElectronFloatingPointValues_.size() != 14) {
      throwWithMessageFromConstructor(
          "test configuration error, expectedElectronFloatingPointValues must have size 14");
    }
    if (expectedElectronIntegralValues_.size() != 2) {
      throwWithMessageFromConstructor("test configuration error, expectedElectronIntegralValues must have size 2");
    }
    if (expectedMuonFloatingPointValues_.size() != 23) {
      throwWithMessageFromConstructor("test configuration error, expectedMuonFloatingPointValues must have size 23");
    }
    if (expectedMuonIntegralValues_.size() != 8) {
      throwWithMessageFromConstructor("test configuration error, expectedMuonIntegralValues must have size 8");
    }
    if (expectedParticleFloatingPointValues_.size() != 4) {
      throwWithMessageFromConstructor("test configuration error, expectedParticleFloatingPointValues must have size 4");
    }
    if (expectedParticleIntegralValues_.size() != 2) {
      throwWithMessageFromConstructor("test configuration error, expectedParticleIntegralValues must have size 2");
    }
    if (expectedPFJetFloatingPointValues_.size() != 15) {
      throwWithMessageFromConstructor("test configuration error, expectedPFJetFloatingPointValues must have size 15");
    }
    if (expectedPFJetIntegralValues_.size() != 8) {
      throwWithMessageFromConstructor("test configuration error, expectedPFJetIntegralValues must have size 8");
    }
    if (expectedPhotonFloatingPointValues_.size() != 8) {
      throwWithMessageFromConstructor("test configuration error, expectedPhotonFloatingPointValues must have size 8");
    }
    if (expectedTrackFloatingPointValues_.size() != 16) {
      throwWithMessageFromConstructor("test configuration error, expectedTrackFloatingPointValues must have size 16");
    }
    if (expectedTrackIntegralValues_.size() != 4) {
      throwWithMessageFromConstructor("test configuration error, expectedTrackIntegralValues must have size 4");
    }
    if (expectedVertexFloatingPointValues_.size() != 7) {
      throwWithMessageFromConstructor("test configuration error, expectedVertexFloatingPointValues must have size 7");
    }
    if (expectedVertexIntegralValues_.size() != 3) {
      throwWithMessageFromConstructor("test configuration error, expectedVertexIntegralValues must have size 3");
    }
  }

  void TestReadRun2Scouting::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    analyzeCaloJets(iEvent);
    analyzeElectrons(iEvent);
    analyzeMuons(iEvent);
    analyzeParticles(iEvent);
    analyzePFJets(iEvent);
    analyzePhotons(iEvent);
    analyzeTracks(iEvent);
    analyzeVertexes(iEvent);
  }

  void TestReadRun2Scouting::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<double>>("expectedCaloJetsValues");
    desc.add<edm::InputTag>("caloJetsTag");
    desc.add<std::vector<double>>("expectedElectronFloatingPointValues");
    desc.add<std::vector<int>>("expectedElectronIntegralValues");
    desc.add<edm::InputTag>("electronsTag");
    desc.add<int>("muonClassVersion");
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
    desc.add<edm::InputTag>("photonsTag");
    desc.add<int>("trackClassVersion");
    desc.add<std::vector<double>>("expectedTrackFloatingPointValues");
    desc.add<std::vector<int>>("expectedTrackIntegralValues");
    desc.add<edm::InputTag>("tracksTag");
    desc.add<int>("vertexClassVersion");
    desc.add<std::vector<double>>("expectedVertexFloatingPointValues");
    desc.add<std::vector<int>>("expectedVertexIntegralValues");
    desc.add<edm::InputTag>("vertexesTag");
    descriptions.addDefault(desc);
  }

  void TestReadRun2Scouting::analyzeCaloJets(edm::Event const& iEvent) const {
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

  void TestReadRun2Scouting::analyzeElectrons(edm::Event const& iEvent) const {
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
        throwWithMessage("analyzeElectrons, dz does not equal expected value");
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
      ++i;
    }
  }

  void TestReadRun2Scouting::analyzeMuons(edm::Event const& iEvent) const {
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
      if (muon.ecalIso() != expectedMuonFloatingPointValues_[4] + offset) {
        throwWithMessage("analyzeMuons, ecalIso does not equal expected value");
      }
      if (muon.hcalIso() != expectedMuonFloatingPointValues_[5] + offset) {
        throwWithMessage("analyzeMuons, hcalIso does not equal expected value");
      }
      if (muon.trackIso() != expectedMuonFloatingPointValues_[6] + offset) {
        throwWithMessage("analyzeMuons, trackIso does not equal expected value");
      }
      if (muon.chi2() != expectedMuonFloatingPointValues_[7] + offset) {
        throwWithMessage("analyzeMuons, chi2 does not equal expected value");
      }
      if (muon.ndof() != expectedMuonFloatingPointValues_[8] + offset) {
        throwWithMessage("analyzeMuons, ndof does not equal expected value");
      }
      if (muon.charge() != expectedMuonIntegralValues_[0] + iOffset) {
        throwWithMessage("analyzeMuons, charge does not equal expected value");
      }
      if (muon.dxy() != expectedMuonFloatingPointValues_[9] + offset) {
        throwWithMessage("analyzeMuons, dxy does not equal expected value");
      }
      if (muon.dz() != expectedMuonFloatingPointValues_[10] + offset) {
        throwWithMessage("analyzeMuons, dz does not equal expected value");
      }
      if (muon.nValidMuonHits() != expectedMuonIntegralValues_[1] + iOffset) {
        throwWithMessage("analyzeMuons, nValidMuonHits does not equal expected value");
      }
      if (muon.nValidPixelHits() != expectedMuonIntegralValues_[2] + iOffset) {
        throwWithMessage("analyzeMuons, nValidPixelHits does not equal expected value");
      }
      if (muon.nMatchedStations() != expectedMuonIntegralValues_[3] + iOffset) {
        throwWithMessage("analyzeMuons, nMatchedStations does not equal expected value");
      }
      if (muon.nTrackerLayersWithMeasurement() != expectedMuonIntegralValues_[4] + iOffset) {
        throwWithMessage("analyzeMuons, nTrackerLayersWithMeasurement does not equal expected value");
      }
      if (muon.type() != expectedMuonIntegralValues_[5] + iOffset) {
        throwWithMessage("analyzeMuons, type does not equal expected value");
      }

      if (inputMuonClassVersion_ > 2) {
        if (muon.nValidStripHits() != expectedMuonIntegralValues_[6] + iOffset) {
          throwWithMessage("analyzeMuons, nValidStripHits does not equal expected value");
        }
        if (muon.trk_qoverp() != expectedMuonFloatingPointValues_[11] + offset) {
          throwWithMessage("analyzeMuons, trk_qoverp does not equal expected value");
        }
        if (muon.trk_lambda() != expectedMuonFloatingPointValues_[12] + offset) {
          throwWithMessage("analyzeMuons, trk_lambda does not equal expected value");
        }
        if (muon.trk_pt() != expectedMuonFloatingPointValues_[13] + offset) {
          throwWithMessage("analyzeMuons, trk_pt does not equal expected value");
        }
        if (muon.trk_phi() != expectedMuonFloatingPointValues_[14] + offset) {
          throwWithMessage("analyzeMuons, trk_phi does not equal expected value");
        }
        if (muon.trk_eta() != expectedMuonFloatingPointValues_[15] + offset) {
          throwWithMessage("analyzeMuons, trk_eta does not equal expected value");
        }
        if (muon.dxyError() != expectedMuonFloatingPointValues_[16] + offset) {
          throwWithMessage("analyzeMuons, dxyError does not equal expected value");
        }
        if (muon.dzError() != expectedMuonFloatingPointValues_[17] + offset) {
          throwWithMessage("analyzeMuons, dzError does not equal expected value");
        }
        if (muon.trk_qoverpError() != expectedMuonFloatingPointValues_[18] + offset) {
          throwWithMessage("analyzeMuons, trk_qoverpError does not equal expected value");
        }
        if (muon.trk_lambdaError() != expectedMuonFloatingPointValues_[19] + offset) {
          throwWithMessage("analyzeMuons, trk_lambdaError does not equal expected value");
        }
        if (muon.trk_phiError() != expectedMuonFloatingPointValues_[20] + offset) {
          throwWithMessage("analyzeMuons, trk_phiError does not equal expected value");
        }
        if (muon.trk_dsz() != expectedMuonFloatingPointValues_[21] + offset) {
          throwWithMessage("analyzeMuons, trk_dsz does not equal expected value");
        }
        if (muon.trk_dszError() != expectedMuonFloatingPointValues_[22] + offset) {
          throwWithMessage("analyzeMuons, trk_dszError does not equal expected value");
        }
        int j = 0;
        for (auto const& val : muon.vtxIndx()) {
          if (val != expectedMuonIntegralValues_[7] + iOffset + 10 * j) {
            throwWithMessage("analyzeMuons, vtxIndx does not contain expected value");
          }
          ++j;
        }
      }
      ++i;
    }
  }

  void TestReadRun2Scouting::analyzeParticles(edm::Event const& iEvent) const {
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
      if (particle.m() != expectedParticleFloatingPointValues_[3] + offset) {
        throwWithMessage("analyzeParticles, m does not equal expected value");
      }
      if (particle.pdgId() != expectedParticleIntegralValues_[0] + iOffset) {
        throwWithMessage("analyzeParticles, pdgId does not equal expected value");
      }
      if (particle.vertex() != expectedParticleIntegralValues_[1] + iOffset) {
        throwWithMessage("analyzeParticles, vertex does not equal expected value");
      }
      ++i;
    }
  }

  void TestReadRun2Scouting::analyzePFJets(edm::Event const& iEvent) const {
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

  void TestReadRun2Scouting::analyzePhotons(edm::Event const& iEvent) const {
    auto const& photons = iEvent.get(photonsToken_);
    unsigned int vectorSize = 2 + iEvent.id().event() % 4;
    if (photons.size() != vectorSize) {
      throwWithMessage("analyzePhotons, photons does not have expected size");
    }
    unsigned int i = 0;
    for (auto const& photon : photons) {
      double offset = static_cast<double>(iEvent.id().event() + i);

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
      ++i;
    }
  }

  void TestReadRun2Scouting::analyzeTracks(edm::Event const& iEvent) const {
    if (inputTrackClassVersion_ < 2) {
      return;
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
      ++i;
    }
  }

  void TestReadRun2Scouting::analyzeVertexes(edm::Event const& iEvent) const {
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
      if (inputVertexClassVersion_ > 2) {
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
      }
      ++i;
    }
  }

  void TestReadRun2Scouting::throwWithMessageFromConstructor(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadRun2Scouting constructor, " << msg;
  }

  void TestReadRun2Scouting::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadRun2Scouting::analyze, " << msg;
  }

}  // namespace edmtest

using edmtest::TestReadRun2Scouting;
DEFINE_FWK_MODULE(TestReadRun2Scouting);
