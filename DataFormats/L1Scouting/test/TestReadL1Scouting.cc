#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
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

#include <memory>
#include <utility>
#include <vector>

namespace edmtest {
  using namespace l1ScoutingRun3;
  class TestReadL1Scouting : public edm::global::EDAnalyzer<> {
  public:
    TestReadL1Scouting(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void analyzeMuons(edm::Event const& iEvent) const;
    void analyzeJets(edm::Event const& iEvent) const;
    void analyzeEGammas(edm::Event const& iEvent) const;
    void analyzeTaus(edm::Event const& iEvent) const;
    void analyzeBxSums(edm::Event const& iEvent) const;

    void throwWithMessageFromConstructor(const char*) const;
    void throwWithMessage(const char*) const;

    const std::vector<unsigned int> bxValues_;

    const std::vector<int> expectedMuonValues_;
    const edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::Muon>> muonsToken_;

    const std::vector<int> expectedJetValues_;
    const edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::Jet>> jetsToken_;

    const std::vector<int> expectedEGammaValues_;
    const edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::EGamma>> eGammasToken_;

    const std::vector<int> expectedTauValues_;
    const edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::Tau>> tausToken_;

    const std::vector<int> expectedBxSumsValues_;
    const edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::BxSums>> bxSumsToken_;
  };

  TestReadL1Scouting::TestReadL1Scouting(edm::ParameterSet const& iPSet)
      : bxValues_(iPSet.getParameter<std::vector<unsigned>>("bxValues")),
        expectedMuonValues_(iPSet.getParameter<std::vector<int>>("expectedMuonValues")),
        muonsToken_(consumes(iPSet.getParameter<edm::InputTag>("muonsTag"))),
        expectedJetValues_(iPSet.getParameter<std::vector<int>>("expectedJetValues")),
        jetsToken_(consumes(iPSet.getParameter<edm::InputTag>("jetsTag"))),
        expectedEGammaValues_(iPSet.getParameter<std::vector<int>>("expectedEGammaValues")),
        eGammasToken_(consumes(iPSet.getParameter<edm::InputTag>("eGammasTag"))),
        expectedTauValues_(iPSet.getParameter<std::vector<int>>("expectedTauValues")),
        tausToken_(consumes(iPSet.getParameter<edm::InputTag>("tausTag"))),
        expectedBxSumsValues_(iPSet.getParameter<std::vector<int>>("expectedBxSumsValues")),
        bxSumsToken_(consumes(iPSet.getParameter<edm::InputTag>("bxSumsTag"))) {
    if (bxValues_.size() != 2) {
      throwWithMessageFromConstructor("bxValues must have 2 elements and it does not");
    }
    if (expectedMuonValues_.size() != 3) {
      throwWithMessageFromConstructor("muonValues must have 3 elements and it does not");
    }
    if (expectedJetValues_.size() != 4) {
      throwWithMessageFromConstructor("jetValues must have 4 elements and it does not");
    }
    if (expectedEGammaValues_.size() != 3) {
      throwWithMessageFromConstructor("eGammaValues must have 3 elements and it does not");
    }
    if (expectedTauValues_.size() != 2) {
      throwWithMessageFromConstructor("tauValues must have 2 elements and it does not");
    }
    if (expectedBxSumsValues_.size() != 1) {
      throwWithMessageFromConstructor("bxSumsValues_ must have 1 elements and it does not");
    }
  }

  void TestReadL1Scouting::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    analyzeMuons(iEvent);
    analyzeJets(iEvent);
    analyzeEGammas(iEvent);
    analyzeTaus(iEvent);
    analyzeBxSums(iEvent);
  }

  void TestReadL1Scouting::analyzeMuons(edm::Event const& iEvent) const {
    auto const& muonsCollection = iEvent.get(muonsToken_);

    for (const unsigned& bx : bxValues_) {
      unsigned nMuons = muonsCollection.getBxSize(bx);
      if (nMuons != expectedMuonValues_.size()) {
        throwWithMessage("analyzeMuons, muons do not have the expected bx size");
      }

      const auto& muons = muonsCollection.bxIterator(bx);
      for (unsigned i = 0; i < nMuons; i++) {
        if (muons[i].hwPt() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwPt does not match the expected value");
        }
        if (muons[i].hwEta() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwEta does not match the expected value");
        }
        if (muons[i].hwPhi() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwPhi does not match the expected value");
        }
        if (muons[i].hwQual() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwQual does not match the expected value");
        }
        if (muons[i].hwCharge() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwCharge does not match the expected value");
        }
        if (muons[i].hwChargeValid() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwChargeValid does not match the expected value");
        }
        if (muons[i].hwIso() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwIso does not match the expected value");
        }
        if (muons[i].hwIndex() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwIndex does not match the expected value");
        }
        if (muons[i].hwEtaAtVtx() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwEtaAtVtx does not match the expected value");
        }
        if (muons[i].hwPhiAtVtx() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwPhiAtVtx does not match the expected value");
        }
        if (muons[i].hwPtUnconstrained() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwPtUnconstrained does not match the expected value");
        }
        if (muons[i].hwDXY() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, hwDXY does not match the expected value");
        }
        if (muons[i].tfMuonIndex() != expectedMuonValues_[i]) {
          throwWithMessage("analyzeMuons, tfMuonIndex does not match the expected value");
        }
      }
    }
  }

  void TestReadL1Scouting::analyzeJets(edm::Event const& iEvent) const {
    auto const& jetsCollection = iEvent.get(jetsToken_);

    for (const unsigned& bx : bxValues_) {
      unsigned nJets = jetsCollection.getBxSize(bx);
      if (nJets != expectedJetValues_.size()) {
        throwWithMessage("analyzeJets, jets do not have the expected bx size");
      }

      const auto& jets = jetsCollection.bxIterator(bx);
      for (unsigned i = 0; i < nJets; i++) {
        if (jets[i].hwEt() != expectedJetValues_[i]) {
          throwWithMessage("analyzeJets, hwEt does not match the expected value");
        }
        if (jets[i].hwEta() != expectedJetValues_[i]) {
          throwWithMessage("analyzeJets, hwEta does not match the expected value");
        }
        if (jets[i].hwPhi() != expectedJetValues_[i]) {
          throwWithMessage("analyzeJets, hwPhi does not match the expected value");
        }
        if (jets[i].hwIso() != expectedJetValues_[i]) {
          throwWithMessage("analyzeJets, hwIso does not match the expected value");
        }
      }
    }
  }

  void TestReadL1Scouting::analyzeEGammas(edm::Event const& iEvent) const {
    auto const& eGammasCollection = iEvent.get(eGammasToken_);

    for (const unsigned& bx : bxValues_) {
      unsigned nEGammas = eGammasCollection.getBxSize(bx);
      if (nEGammas != expectedEGammaValues_.size()) {
        throwWithMessage("analyzeEGammas, egammas do not have the expected bx size");
      }

      const auto& eGammas = eGammasCollection.bxIterator(bx);
      for (unsigned i = 0; i < nEGammas; i++) {
        if (eGammas[i].hwEt() != expectedEGammaValues_[i]) {
          throwWithMessage("analyzeEGammas, hwEt does not match the expected value");
        }
        if (eGammas[i].hwEta() != expectedEGammaValues_[i]) {
          throwWithMessage("analyzeEGammas, hwEta does not match the expected value");
        }
        if (eGammas[i].hwPhi() != expectedEGammaValues_[i]) {
          throwWithMessage("analyzeEGammas, hwPhi does not match the expected value");
        }
        if (eGammas[i].hwIso() != expectedEGammaValues_[i]) {
          throwWithMessage("analyzeEGammas, hwIso does not match the expected value");
        }
      }
    }
  }

  void TestReadL1Scouting::analyzeTaus(edm::Event const& iEvent) const {
    auto const& tausCollection = iEvent.get(tausToken_);

    for (const unsigned& bx : bxValues_) {
      unsigned nTaus = tausCollection.getBxSize(bx);
      if (nTaus != expectedTauValues_.size()) {
        throwWithMessage("analyzeTaus, taus do not have the expected bx size");
      }

      const auto& taus = tausCollection.bxIterator(bx);
      for (unsigned i = 0; i < nTaus; i++) {
        if (taus[i].hwEt() != expectedTauValues_[i]) {
          throwWithMessage("analyzeTaus, hwEt does not match the expected value");
        }
        if (taus[i].hwEta() != expectedTauValues_[i]) {
          throwWithMessage("analyzeTaus, hwEta does not match the expected value");
        }
        if (taus[i].hwPhi() != expectedTauValues_[i]) {
          throwWithMessage("analyzeTaus, hwPhi does not match the expected value");
        }
        if (taus[i].hwIso() != expectedTauValues_[i]) {
          throwWithMessage("analyzeTaus, hwIso does not match the expected value");
        }
      }
    }
  }

  void TestReadL1Scouting::analyzeBxSums(edm::Event const& iEvent) const {
    auto const& bxSumsCollection = iEvent.get(bxSumsToken_);

    for (const unsigned& bx : bxValues_) {
      unsigned nSums = bxSumsCollection.getBxSize(bx);
      if (nSums != expectedBxSumsValues_.size()) {
        throwWithMessage("analyzeBxSums, sums do not have the expected bx size");
      }

      const auto& sums = bxSumsCollection.bxIterator(bx);
      for (unsigned i = 0; i < nSums; i++) {
        if (sums[i].hwTotalEt() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwTotalEt does not match the expected value");
        }
        if (sums[i].hwTotalEtEm() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwTotalEtEm does not match the expected value");
        }
        if (sums[i].hwTotalHt() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwTotalHt does not match the expected value");
        }
        if (sums[i].hwMissEt() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwMissEt does not match the expected value");
        }
        if (sums[i].hwMissEtPhi() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwMissEtPhi does not match the expected value");
        }
        if (sums[i].hwMissHt() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwMissHt does not match the expected value");
        }
        if (sums[i].hwMissHtPhi() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwMissHtPhi does not match the expected value");
        }
        if (sums[i].hwMissEtHF() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwMissEtHF does not match the expected value");
        }
        if (sums[i].hwMissEtHFPhi() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwMissEtHFPhi does not match the expected value");
        }
        if (sums[i].hwMissHtHF() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwMissHtHFPhi does not match the expected value");
        }
        if (sums[i].hwAsymEt() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwAsymEt does not match the expected value");
        }
        if (sums[i].hwAsymHt() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwAsymHt does not match the expected value");
        }
        if (sums[i].hwAsymEtHF() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwAsymEtHF does not match the expected value");
        }
        if (sums[i].hwAsymHtHF() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, hwAsymHtHF does not match the expected value");
        }
        if (sums[i].minBiasHFP0() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, minBiasHFP0 does not match the expected value");
        }
        if (sums[i].minBiasHFM0() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, minBiasHFM0 does not match the expected value");
        }
        if (sums[i].minBiasHFP1() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, minBiasHFP1 does not match the expected value");
        }
        if (sums[i].minBiasHFM1() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, minBiasHFM1 does not match the expected value");
        }
        if (sums[i].towerCount() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, towerCount does not match the expected value");
        }
        if (sums[i].centrality() != expectedBxSumsValues_[i]) {
          throwWithMessage("analyzeBxSums, centrality does not match the expected value");
        }
      }
    }
  }

  void TestReadL1Scouting::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<unsigned int>>("bxValues");
    desc.add<std::vector<int>>("expectedMuonValues");
    desc.add<edm::InputTag>("muonsTag");
    desc.add<std::vector<int>>("expectedJetValues");
    desc.add<edm::InputTag>("jetsTag");
    desc.add<std::vector<int>>("expectedEGammaValues");
    desc.add<edm::InputTag>("eGammasTag");
    desc.add<std::vector<int>>("expectedTauValues");
    desc.add<edm::InputTag>("tausTag");
    desc.add<std::vector<int>>("expectedBxSumsValues");
    desc.add<edm::InputTag>("bxSumsTag");
    descriptions.addDefault(desc);
  }

  void TestReadL1Scouting::throwWithMessageFromConstructor(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadL1Scouting constructor, test configuration error, " << msg;
  }

  void TestReadL1Scouting::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestReadL1Scouting analyzer, " << msg;
  }

}  // namespace edmtest

using edmtest::TestReadL1Scouting;
DEFINE_FWK_MODULE(TestReadL1Scouting);