#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondFormats/GeometryObjects/interface/EcalSimulationParameters.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include <iostream>

class EcalSimParametersAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalSimParametersAnalyzer(const edm::ParameterSet&);

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  edm::ESGetToken<EcalSimulationParameters, IdealGeometryRecord> simparToken_;
  std::string name_;
};

EcalSimParametersAnalyzer::EcalSimParametersAnalyzer(const edm::ParameterSet& ic)
    : name_(ic.getUntrackedParameter<std::string>("name")) {
  simparToken_ = esConsumes<EcalSimulationParameters, IdealGeometryRecord>(edm::ESInputTag{"", name_});
}

void EcalSimParametersAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("name", "EcalHitsEB");
  descriptions.add("ecalSimulationParametersAnalyzerEB", desc);
}

void EcalSimParametersAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const auto& parS = iSetup.getData(simparToken_);
  const EcalSimulationParameters* parsim = &parS;
  if (parsim != nullptr) {
    std::cout << "\n nxtalEta_: " << parsim->nxtalEta_;
    std::cout << "\n nxtalPhi_: " << parsim->nxtalPhi_;
    std::cout << "\n phiBaskets_: " << parsim->phiBaskets_;
    std::cout << "\n etaBaskets_: ";
    for (const auto& it : parsim->etaBaskets_)
      std::cout << it << ", ";
    std::cout << "\n ncrys_: " << parsim->ncrys_;
    std::cout << "\n nmods_: " << parsim->nmods_;
    std::cout << "\n useWeight_: " << parsim->useWeight_;
    std::cout << "\n depth1Name_: " << parsim->depth1Name_;
    std::cout << "\n depth2Name_: " << parsim->depth2Name_;
    std::cout << "\n lvNames_: " << parsim->lvNames_.size() << " occurences\n";
    int kount(0);
    for (const auto& it : parsim->lvNames_) {
      std::cout << it << ", ";
      ++kount;
      if (kount == 8) {
        std::cout << "\n";
        kount = 0;
      }
    }
    kount = 0;
    std::cout << "\n matNames_: " << parsim->matNames_.size() << " occurences\n";
    for (const auto& it : parsim->matNames_) {
      std::cout << it << ", ";
      ++kount;
      if (kount == 7) {
        std::cout << "\n";
        kount = 0;
      }
    }
    kount = 0;
    std::cout << "\n dzs_: " << parsim->dzs_.size() << " occurences\n";
    for (const auto& it : parsim->dzs_) {
      std::cout << it << ", ";
      ++kount;
      if (kount == 20) {
        std::cout << "\n";
        kount = 0;
      }
    }
    std::cout << std::endl;
  }
}

DEFINE_FWK_MODULE(EcalSimParametersAnalyzer);
