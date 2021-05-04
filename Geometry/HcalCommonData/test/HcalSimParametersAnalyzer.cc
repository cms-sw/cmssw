#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include <iostream>

class HcalSimParametersAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HcalSimParametersAnalyzer(const edm::ParameterSet&);

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  edm::ESGetToken<HcalSimulationParameters, HcalParametersRcd> simparToken_;
};

HcalSimParametersAnalyzer::HcalSimParametersAnalyzer(const edm::ParameterSet&) {
  simparToken_ = esConsumes<HcalSimulationParameters, HcalParametersRcd>(edm::ESInputTag{});
}

void HcalSimParametersAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const auto& parS = iSetup.getData(simparToken_);
  const HcalSimulationParameters* parsim = &parS;
  if (parsim != nullptr) {
    std::cout << "\nattenuationLength_: ";
    for (const auto& it : parsim->attenuationLength_)
      std::cout << it << ", ";
    std::cout << "\nlambdaLimits_: ";
    for (const auto& it : parsim->lambdaLimits_)
      std::cout << it << ", ";
    std::cout << "\nshortFiberLength_: ";
    for (const auto& it : parsim->shortFiberLength_)
      std::cout << it << ", ";
    std::cout << "\nlongFiberLength_: ";
    for (const auto& it : parsim->longFiberLength_)
      std::cout << it << ", ";
    std::cout << std::endl;

    std::cout << "\npmtRight_: ";
    for (const auto& it : parsim->pmtRight_)
      std::cout << it << ", ";
    std::cout << "\npmtFiberRight_: ";
    for (const auto& it : parsim->pmtFiberRight_)
      std::cout << it << ", ";
    std::cout << "\npmtLeft_: ";
    for (const auto& it : parsim->pmtLeft_)
      std::cout << it << ", ";
    std::cout << "\npmtFiberLeft_: ";
    for (const auto& it : parsim->pmtFiberLeft_)
      std::cout << it << ", ";
    std::cout << std::endl;

    std::cout << "\nhfLevels_: ";
    for (const auto& it : parsim->hfLevels_)
      std::cout << it << ", ";
    std::cout << "\nhfNames_: ";
    for (const auto& it : parsim->hfNames_)
      std::cout << it << ", ";
    std::cout << "\nhfFibreNames_: ";
    for (const auto& it : parsim->hfFibreNames_)
      std::cout << it << ", ";
    std::cout << "\nhfPMTNames_: ";
    for (const auto& it : parsim->hfPMTNames_)
      std::cout << it << ", ";
    std::cout << "\nhfFibreStraightNames_: ";
    for (const auto& it : parsim->hfFibreStraightNames_)
      std::cout << it << ", ";
    std::cout << "\nhfFibreConicalNames_: ";
    for (const auto& it : parsim->hfFibreConicalNames_)
      std::cout << it << ", ";
    std::cout << "\nhcalMaterialNames_: ";
    for (const auto& it : parsim->hcalMaterialNames_)
      std::cout << it << ", ";
    std::cout << std::endl;
  }
}

DEFINE_FWK_MODULE(HcalSimParametersAnalyzer);
