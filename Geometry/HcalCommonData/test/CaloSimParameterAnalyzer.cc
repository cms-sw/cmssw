#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/GeometryObjects/interface/CaloSimulationParameters.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include <iostream>
#include <sstream>

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
    std::ostringstream st1;
    st1 << "\ncaloNames_: ";
    for (const auto& it : parsim->caloNames_)
      st1 << it << ", ";
    st1 << "\nlevels_: ";
    for (const auto& it : parsim->levels_)
      st1 << it << ", ";
    st1 << "\nneighbours_: ";
    for (const auto& it : parsim->neighbours_)
      st1 << it << ", ";
    st1 << "\ninsideNames_: ";
    for (const auto& it : parsim->insideNames_)
      st1 << it << ", ";
    edm::LogVerbatim("HCalGeom") << st1.str();

    std::ostringstream st2;
    st2 << "\nfCaloNames_: ";
    for (const auto& it : parsim->fCaloNames_)
      st2 << it << ", ";
    st2 << "\nfLevels_: ";
    for (const auto& it : parsim->fLevels_)
      st2 << it << ", ";
    edm::LogVerbatim("HCalGeom") << st2.str();
  }
}

DEFINE_FWK_MODULE(CaloSimParametersAnalyzer);
