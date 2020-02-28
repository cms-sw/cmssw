#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/GeometryObjects/interface/CaloSimulationParameters.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include <iostream>

class CaloSimParametersAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit CaloSimParametersAnalyzer(const edm::ParameterSet&);
  ~CaloSimParametersAnalyzer(void) override;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  edm::ESGetToken<CaloSimulationParameters, HcalParametersRcd> simparToken_;
};

CaloSimParametersAnalyzer::CaloSimParametersAnalyzer(const edm::ParameterSet&) {
  simparToken_ = esConsumes<CaloSimulationParameters, HcalParametersRcd>(edm::ESInputTag{});
}

CaloSimParametersAnalyzer::~CaloSimParametersAnalyzer(void) {}

void CaloSimParametersAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const auto& parS = iSetup.getData(simparToken_);
  const CaloSimulationParameters* parsim = &parS;
  if (parsim != nullptr) {
    std::cout << "\ncaloNames_: ";
    for (const auto& it : parsim->caloNames_)
      std::cout << it << ", ";
    std::cout << "\nlevels_: ";
    for (const auto& it : parsim->levels_)
      std::cout << it << ", ";
    std::cout << "\nneighbours_: ";
    for (const auto& it : parsim->neighbours_)
      std::cout << it << ", ";
    std::cout << "\ninsideNames_: ";
    for (const auto& it : parsim->insideNames_)
      std::cout << it << ", ";
    std::cout << std::endl;

    std::cout << "\nfCaloNames_: ";
    for (const auto& it : parsim->fCaloNames_)
      std::cout << it << ", ";
    std::cout << "\nfLevels_: ";
    for (const auto& it : parsim->fLevels_)
      std::cout << it << ", ";
    std::cout << std::endl;
  }
}

DEFINE_FWK_MODULE(CaloSimParametersAnalyzer);
