#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include <iostream>
#include <sstream>

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
    std::ostringstream st1;
    st1 << "\nattenuationLength_: ";
    for (const auto& it : parsim->attenuationLength_)
      st1 << it << ", ";
    st1 << "\nlambdaLimits_: ";
    for (const auto& it : parsim->lambdaLimits_)
      st1 << it << ", ";
    st1 << "\nshortFiberLength_: ";
    for (const auto& it : parsim->shortFiberLength_)
      st1 << it << ", ";
    st1 << "\nlongFiberLength_: ";
    for (const auto& it : parsim->longFiberLength_)
      st1 << it << ", ";
    edm::LogVerbatim("HCalGeom") << st1.str();

    std::ostringstream st2;
    st2 << "\npmtRight_: ";
    for (const auto& it : parsim->pmtRight_)
      st2 << it << ", ";
    st2 << "\npmtFiberRight_: ";
    for (const auto& it : parsim->pmtFiberRight_)
      st2 << it << ", ";
    st2 << "\npmtLeft_: ";
    for (const auto& it : parsim->pmtLeft_)
      st2 << it << ", ";
    st2 << "\npmtFiberLeft_: ";
    for (const auto& it : parsim->pmtFiberLeft_)
      st2 << it << ", ";
    edm::LogVerbatim("HCalGeom") << st2.str();

    std::ostringstream st3;
    st3 << "\nhfLevels_: ";
    for (const auto& it : parsim->hfLevels_)
      st3 << it << ", ";
    st3 << "\nhfNames_: ";
    for (const auto& it : parsim->hfNames_)
      st3 << it << ", ";
    st3 << "\nhfFibreNames_: ";
    for (const auto& it : parsim->hfFibreNames_)
      st3 << it << ", ";
    st3 << "\nhfPMTNames_: ";
    for (const auto& it : parsim->hfPMTNames_)
      st3 << it << ", ";
    st3 << "\nhfFibreStraightNames_: ";
    for (const auto& it : parsim->hfFibreStraightNames_)
      st3 << it << ", ";
    st3 << "\nhfFibreConicalNames_: ";
    for (const auto& it : parsim->hfFibreConicalNames_)
      st3 << it << ", ";
    st3 << "\nhcalMaterialNames_: ";
    for (const auto& it : parsim->hcalMaterialNames_)
      st3 << it << ", ";
    edm::LogVerbatim("HCalGeom") << st3.str();
  }
}

DEFINE_FWK_MODULE(HcalSimParametersAnalyzer);
