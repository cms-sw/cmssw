#include "DataFormats/L1Scouting/interface/L1ScoutingBMTFStub.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCaloTower.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>
#include <utility>
#include <vector>

namespace edmtest {
  using namespace l1ScoutingRun3;
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

    const std::vector<int> caloTowersValues_;
    const edm::EDPutTokenT<OrbitCollection<l1ScoutingRun3::CaloTower>> caloTowersPutToken_;
  };

  TestWriteL1Scouting::TestWriteL1Scouting(edm::ParameterSet const& iPSet)
      : bxValues_(iPSet.getParameter<std::vector<unsigned>>("bxValues")),
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
        bmtfStubsPutToken_(produces()) 
        caloTowersValues_(iPSet.getParameter<std::vector<int>>("caloTowersValues")),
        caloTowersPutToken_(produces()) {
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
      throwWithMessage("bxSumsValues_ must have 1 elements and it does not");
    }
    if (bmtfStubsValues_.size() != 2) {
      throwWithMessage("bmtfStubsValues_ must have 2 elements and it does not");
    }
    if (caloTowersValues_.size() != 2) {
      throwWithMessage("caloTowersValues_ must have 2 elements and it does not");
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

  }

  void TestWriteL1Scouting::produceMuons(edm::Event& iEvent) const {
    std::unique_ptr<l1ScoutingRun3::MuonOrbitCollection> muons(new l1ScoutingRun3::MuonOrbitCollection);

    std::vector<std::vector<l1ScoutingRun3::Muon>> orbitBufferMuons(3565);
    int nMuons = 0;
    for (const unsigned& bx : bxValues_) {
      for (const int& val : muonValues_) {
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
    for (const unsigned& bx : bxValues_) {
      for (const int& val : jetValues_) {
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
    for (const unsigned& bx : bxValues_) {
      for (const int& val : eGammaValues_) {
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
    for (const unsigned& bx : bxValues_) {
      for (const int& val : tauValues_) {
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
    for (const unsigned& bx : bxValues_) {
      for (const int& val : bxSumsValues_) {
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
    for (const unsigned& bx : bxValues_) {
      for (const int& val : bmtfStubsValues_) {
        orbitBufferStubs[bx].emplace_back(val + 8, val + 7, val + 6, val + 5, val + 4, val + 3, val + 2, val + 1, val);
        nStubs++;
      }
    }

    stubs->fillAndClear(orbitBufferStubs, nStubs);
    iEvent.put(bmtfStubsPutToken_, std::move(stubs));
  }

  void TestWriteL1Scouting::produceCaloTowers(edm::Event& iEvent) const {
    std::unique_ptr<l1ScoutingRun3::CaloTowerOrbitCollection> caloTowers(new l1ScoutingRun3::CaloTowerOrbitCollection);

    std::vector<std::vector<l1ScoutingRun3::CaloTower>> orbitBufferCaloTowers(3565);
    int nCaloTowers = 0;
    for (const unsigned& bx : bxValues_) {
      for (const int& val : caloTowerValues_) {
        orbitBufferCaloTowers[bx].emplace_back(val, val, val, val);
        nCaloTowers++;
      }
    }

    caloTowers->fillAndClear(orbitBufferCaloTowers, nCaloTowers);
    iEvent.put(caloTowersPutToken_, std::move(caloTowers));
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

    descriptions.addDefault(desc);
  }

  void TestWriteL1Scouting::throwWithMessage(const char* msg) const {
    throw cms::Exception("TestFailure") << "TestWriteL1Scouting constructor, test configuration error, " << msg;
  }

}  // namespace edmtest

using edmtest::TestWriteL1Scouting;
DEFINE_FWK_MODULE(TestWriteL1Scouting);
