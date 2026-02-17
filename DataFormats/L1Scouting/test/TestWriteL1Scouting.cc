#include "DataFormats/L1Scouting/interface/L1ScoutingBMTFStub.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCaloTower.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingFastJet.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <utility>
#include <vector>

namespace edmtest {
  class TestWriteL1Scouting : public edm::global::EDProducer<> {
  public:
    TestWriteL1Scouting(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void produceMuons(edm::Event& iEvent) const;
    void produceJets(edm::Event& iEvent) const;
    void produceEGammas(edm::Event& iEvent) const;
    void produceTaus(edm::Event& iEvent) const;
    void produceBxSums(edm::Event& iEvent) const;
    void produceBmtfStubs(edm::Event& iEvent) const;
    void produceCaloTowers(edm::Event& iEvent) const;
    void produceFastJets(edm::Event& iEvent) const;

    void throwWithMessage(const char*) const;

    const std::vector<unsigned int> bxValues_;

    const std::vector<int> muonValues_;
    const edm::EDPutTokenT<OrbitCollection<l1ScoutingRun3::Muon>> muonsPutToken_;

    const std::vector<int> jetValues_;
    const edm::EDPutTokenT<OrbitCollection<l1ScoutingRun3::Jet>> jetsPutToken_;

    const std::vector<int> eGammaValues_;
    const edm::EDPutTokenT<OrbitCollection<l1ScoutingRun3::EGamma>> eGammasPutToken_;

    const std::vector<int> tauValues_;
    const edm::EDPutTokenT<OrbitCollection<l1ScoutingRun3::Tau>> tausPutToken_;

    const std::vector<int> bxSumsValues_;
    const edm::EDPutTokenT<OrbitCollection<l1ScoutingRun3::BxSums>> bxSumsPutToken_;

    const std::vector<int> bmtfStubsValues_;
    const edm::EDPutTokenT<OrbitCollection<l1ScoutingRun3::BMTFStub>> bmtfStubsPutToken_;

    const std::vector<int> caloTowerValues_;
    const edm::EDPutTokenT<OrbitCollection<l1ScoutingRun3::CaloTower>> caloTowersPutToken_;

    const std::vector<double> fastJetFloatingPointValues_;
    const std::vector<int> fastJetIntegralValues_;
    const edm::EDPutTokenT<OrbitCollection<l1ScoutingRun3::FastJet>> fastJetsPutToken_;
  };

  TestWriteL1Scouting::TestWriteL1Scouting(edm::ParameterSet const& iPSet)
      : bxValues_(iPSet.getParameter<std::vector<unsigned int>>("bxValues")),
        muonValues_(iPSet.getParameter<std::vector<int>>("muonValues")),
        muonsPutToken_(produces()),
        jetValues_(iPSet.getParameter<std::vector<int>>("jetValues")),
        jetsPutToken_(produces()),
        eGammaValues_(iPSet.getParameter<std::vector<int>>("eGammaValues")),
        eGammasPutToken_(produces()),
        tauValues_(iPSet.getParameter<std::vector<int>>("tauValues")),
        tausPutToken_(produces()),
        bxSumsValues_(iPSet.getParameter<std::vector<int>>("bxSumsValues")),
        bxSumsPutToken_(produces()),
        bmtfStubsValues_(iPSet.getParameter<std::vector<int>>("bmtfStubValues")),
        bmtfStubsPutToken_(produces()),
        caloTowerValues_(iPSet.getParameter<std::vector<int>>("caloTowerValues")),
        caloTowersPutToken_(produces()),
        fastJetFloatingPointValues_(iPSet.getParameter<std::vector<double>>("fastJetFloatingPointValues")),
        fastJetIntegralValues_(iPSet.getParameter<std::vector<int>>("fastJetIntegralValues")),
        fastJetsPutToken_(produces()) {
    if (bxValues_.size() != 2) {
      throwWithMessage("bxValues must have 2 elements and it does not");
    }
    if (muonValues_.size() != 3) {
      throwWithMessage("muonValues must have 3 elements and it does not");
    }
    if (jetValues_.size() != 4) {
      throwWithMessage("jetValues must have 4 elements and it does not");
    }
    if (eGammaValues_.size() != 3) {
      throwWithMessage("eGammaValues must have 3 elements and it does not");
    }
    if (tauValues_.size() != 2) {
      throwWithMessage("tauValues must have 2 elements and it does not");
    }
    if (bxSumsValues_.size() != 1) {
      throwWithMessage("bxSumsValues must have 1 elements and it does not");
    }
    if (bmtfStubsValues_.size() != 2) {
      throwWithMessage("bmtfStubsValues must have 2 elements and it does not");
    }
    if (caloTowerValues_.size() != 5) {
      throwWithMessage("caloTowerValues must have 5 elements and it does not");
    }
    if (fastJetFloatingPointValues_.size() != 3) {
      throwWithMessage("fastJetFloatingPointValues must have 3 elements and it does not");
    }
    if (fastJetIntegralValues_.size() != 3) {
      throwWithMessage("fastJetIntegralValues must have 3 elements and it does not");
    }
  }

  void TestWriteL1Scouting::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    produceMuons(iEvent);
    produceJets(iEvent);
    produceEGammas(iEvent);
    produceTaus(iEvent);
    produceBxSums(iEvent);
    produceBmtfStubs(iEvent);
    produceCaloTowers(iEvent);
    produceFastJets(iEvent);
  }

  void TestWriteL1Scouting::produceMuons(edm::Event& iEvent) const {
    std::unique_ptr<l1ScoutingRun3::MuonOrbitCollection> muons(new l1ScoutingRun3::MuonOrbitCollection);

    std::vector<std::vector<l1ScoutingRun3::Muon>> orbitBufferMuons(3565);
    int nMuons = 0;
    for (auto const bx : bxValues_) {
      for (auto const val : muonValues_) {
        orbitBufferMuons[bx].emplace_back(val, val, val, val, val, val, val, val, val, val, val, val);
        nMuons++;
      }
    }

    muons->fillAndClear(orbitBufferMuons, nMuons);
    iEvent.put(muonsPutToken_, std::move(muons));
  }

  void TestWriteL1Scouting::produceJets(edm::Event& iEvent) const {
    std::unique_ptr<l1ScoutingRun3::JetOrbitCollection> jets(new l1ScoutingRun3::JetOrbitCollection);

    std::vector<std::vector<l1ScoutingRun3::Jet>> orbitBufferJets(3565);
    int nJets = 0;
    for (auto const bx : bxValues_) {
      for (auto const val : jetValues_) {
        orbitBufferJets[bx].emplace_back(val, val, val, val);
        nJets++;
      }
    }

    jets->fillAndClear(orbitBufferJets, nJets);
    iEvent.put(jetsPutToken_, std::move(jets));
  }

  void TestWriteL1Scouting::produceEGammas(edm::Event& iEvent) const {
    std::unique_ptr<l1ScoutingRun3::EGammaOrbitCollection> eGammas(new l1ScoutingRun3::EGammaOrbitCollection);

    std::vector<std::vector<l1ScoutingRun3::EGamma>> orbitBufferEGammas(3565);
    int nEGammas = 0;
    for (auto const bx : bxValues_) {
      for (auto const val : eGammaValues_) {
        orbitBufferEGammas[bx].emplace_back(val, val, val, val);
        nEGammas++;
      }
    }

    eGammas->fillAndClear(orbitBufferEGammas, nEGammas);
    iEvent.put(eGammasPutToken_, std::move(eGammas));
  }

  void TestWriteL1Scouting::produceTaus(edm::Event& iEvent) const {
    std::unique_ptr<l1ScoutingRun3::TauOrbitCollection> taus(new l1ScoutingRun3::TauOrbitCollection);

    std::vector<std::vector<l1ScoutingRun3::Tau>> orbitBufferTaus(3565);
    int nTaus = 0;
    for (auto const bx : bxValues_) {
      for (auto const val : tauValues_) {
        orbitBufferTaus[bx].emplace_back(val, val, val, val);
        nTaus++;
      }
    }

    taus->fillAndClear(orbitBufferTaus, nTaus);
    iEvent.put(tausPutToken_, std::move(taus));
  }

  void TestWriteL1Scouting::produceBxSums(edm::Event& iEvent) const {
    std::unique_ptr<l1ScoutingRun3::BxSumsOrbitCollection> bxSums(new l1ScoutingRun3::BxSumsOrbitCollection);

    std::vector<std::vector<l1ScoutingRun3::BxSums>> orbitBufferBxSums(3565);
    int nBxSums = 0;
    for (auto const bx : bxValues_) {
      for (auto const val : bxSumsValues_) {
        orbitBufferBxSums[bx].emplace_back(
            val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val);
        nBxSums++;
      }
    }

    bxSums->fillAndClear(orbitBufferBxSums, nBxSums);
    iEvent.put(bxSumsPutToken_, std::move(bxSums));
  }

  void TestWriteL1Scouting::produceBmtfStubs(edm::Event& iEvent) const {
    std::unique_ptr<l1ScoutingRun3::BMTFStubOrbitCollection> stubs(new l1ScoutingRun3::BMTFStubOrbitCollection);

    std::vector<std::vector<l1ScoutingRun3::BMTFStub>> orbitBufferStubs(3565);
    int nStubs = 0;
    for (auto const bx : bxValues_) {
      for (auto const val : bmtfStubsValues_) {
        orbitBufferStubs[bx].emplace_back(val + 8, val + 7, val + 6, val + 5, val + 4, val + 3, val + 2, val + 1, val);
        nStubs++;
      }
    }

    stubs->fillAndClear(orbitBufferStubs, nStubs);
    iEvent.put(bmtfStubsPutToken_, std::move(stubs));
  }

  void TestWriteL1Scouting::produceCaloTowers(edm::Event& iEvent) const {
    auto caloTowers = std::make_unique<l1ScoutingRun3::CaloTowerOrbitCollection>();
    std::vector<std::vector<l1ScoutingRun3::CaloTower>> orbitBufferCaloTowers(3565);
    int nCaloTowers = 0;

    for (auto bx_idx = 0u; bx_idx < bxValues_.size(); ++bx_idx) {
      auto const bx = bxValues_[bx_idx];
      int const val_offset = iEvent.id().event() % 100 + bx_idx;
      for (auto val_idx = 0u; val_idx < caloTowerValues_.size(); ++val_idx) {
        int const val = caloTowerValues_[val_idx] + val_offset;
        orbitBufferCaloTowers[bx].emplace_back(val, val + 1, val + 2, val + 3, val + 4);
        ++nCaloTowers;
      }
    }

    caloTowers->fillAndClear(orbitBufferCaloTowers, nCaloTowers);
    iEvent.put(caloTowersPutToken_, std::move(caloTowers));
  }

  void TestWriteL1Scouting::produceFastJets(edm::Event& iEvent) const {
    auto fastJets = std::make_unique<l1ScoutingRun3::FastJetOrbitCollection>();
    std::vector<std::vector<l1ScoutingRun3::FastJet>> orbitBufferFastJets(3565);
    int nFastJets = 0;

    for (auto bx_idx = 0u; bx_idx < bxValues_.size(); ++bx_idx) {
      auto const bx = bxValues_[bx_idx];
      int const val_offset = iEvent.id().event() % 100 + bx_idx;
      for (auto val_idx = 0u; val_idx < fastJetIntegralValues_.size(); ++val_idx) {
        float const val_flp = fastJetFloatingPointValues_.at(val_idx) + val_offset;
        int const val_int = fastJetIntegralValues_[val_idx] + val_offset;
        orbitBufferFastJets[bx].emplace_back(val_flp, val_flp + 1, val_flp + 2, val_flp + 3, val_int, val_flp + 4);
        ++nFastJets;
      }
    }

    fastJets->fillAndClear(orbitBufferFastJets, nFastJets);
    iEvent.put(fastJetsPutToken_, std::move(fastJets));
  }

  void TestWriteL1Scouting::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<unsigned int>>("bxValues");
    desc.add<std::vector<int>>("muonValues");
    desc.add<std::vector<int>>("jetValues");
    desc.add<std::vector<int>>("eGammaValues");
    desc.add<std::vector<int>>("tauValues");
    desc.add<std::vector<int>>("bxSumsValues");
    desc.add<std::vector<int>>("bmtfStubValues");
    desc.add<std::vector<int>>("caloTowerValues");
    desc.add<std::vector<double>>("fastJetFloatingPointValues");
    desc.add<std::vector<int>>("fastJetIntegralValues");

    descriptions.addDefault(desc);
  }

  void TestWriteL1Scouting::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestWriteL1Scouting constructor, test configuration error, " << msg;
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
using edmtest::TestWriteL1Scouting;
DEFINE_FWK_MODULE(TestWriteL1Scouting);
