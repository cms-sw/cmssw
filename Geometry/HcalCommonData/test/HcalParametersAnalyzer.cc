#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include <sstream>
#include <iostream>

class HcalParametersAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HcalParametersAnalyzer(const edm::ParameterSet&);
  ~HcalParametersAnalyzer(void) override;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  edm::ESGetToken<HcalParameters, HcalParametersRcd> parToken_;
  edm::ESGetToken<HcalSimulationParameters, HcalParametersRcd> simparToken_;
};

HcalParametersAnalyzer::HcalParametersAnalyzer(const edm::ParameterSet&) {
  parToken_ = esConsumes<HcalParameters, HcalParametersRcd>(edm::ESInputTag{});
  simparToken_ = esConsumes<HcalSimulationParameters, HcalParametersRcd>(edm::ESInputTag{});
}

HcalParametersAnalyzer::~HcalParametersAnalyzer(void) {}

void HcalParametersAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const auto& parR = iSetup.getData(parToken_);
  const HcalParameters* pars = &parR;

  std::ostringstream st1;
  st1 << "rHB: ";
  for (const auto& it : pars->rHB)
    st1 << it << ", ";
  st1 << "\ndrHB: ";
  for (const auto& it : pars->drHB)
    st1 << it << ", ";
  st1 << "\nzHE: ";
  for (const auto& it : pars->zHE)
    st1 << it << ", ";
  st1 << "\ndzHE: ";
  for (const auto& it : pars->dzHE)
    st1 << it << ", ";
  st1 << "\nzHO: ";
  for (const auto& it : pars->zHO)
    st1 << it << ", ";
  st1 << "\nrHO: ";
  for (const auto& it : pars->rHO)
    st1 << it << ", ";
  st1 << "\nrhoxHB: ";
  for (const auto& it : pars->rhoxHB)
    st1 << it << ", ";
  st1 << "\nzxHB: ";
  for (const auto& it : pars->zxHB)
    st1 << it << ", ";
  st1 << "\ndyHB: ";
  for (const auto& it : pars->dyHB)
    st1 << it << ", ";
  st1 << "\ndxHB: ";
  for (const auto& it : pars->dxHB)
    st1 << it << ", ";
  st1 << "\nrhoxHE: ";
  for (const auto& it : pars->rhoxHE)
    st1 << it << ", ";
  st1 << "\nzxHE: ";
  for (const auto& it : pars->zxHE)
    st1 << it << ", ";
  st1 << "\ndyHE: ";
  for (const auto& it : pars->dyHE)
    st1 << it << ", ";
  st1 << "\ndx1HE: ";
  for (const auto& it : pars->dx1HE)
    st1 << it << ", ";
  st1 << "\ndx2HE: ";
  for (const auto& it : pars->dx2HE)
    st1 << it << ", ";
  st1 << "\nphioff: ";
  for (const auto& it : pars->phioff)
    st1 << it << ", ";
  st1 << "\netaTable: ";
  for (const auto& it : pars->etaTable)
    st1 << it << ", ";
  st1 << "\nrTable: ";
  for (const auto& it : pars->rTable)
    st1 << it << ", ";
  st1 << "\nphibin: ";
  for (const auto& it : pars->phibin)
    st1 << it << ", ";
  st1 << "\nphitable: ";
  for (const auto& it : pars->phitable)
    st1 << it << ", ";
  st1 << "\netaRange: ";
  for (const auto& it : pars->etaRange)
    st1 << it << ", ";
  st1 << "\ngparHF: ";
  for (const auto& it : pars->gparHF)
    st1 << it << ", ";
  st1 << "\nLayer0Wt: ";
  for (const auto& it : pars->Layer0Wt)
    st1 << it << ", ";
  st1 << "\nHBGains: ";
  for (const auto& it : pars->HBGains)
    st1 << it << ", ";
  st1 << "\nHEGains: ";
  for (const auto& it : pars->HEGains)
    st1 << it << ", ";
  st1 << "\nHFGains: ";
  for (const auto& it : pars->HFGains)
    st1 << it << ", ";
  st1 << "\netaTableHF: ";
  for (const auto& it : pars->etaTableHF)
    st1 << it << ", ";
  st1 << "\nmodHB: ";
  for (const auto& it : pars->modHB)
    st1 << it << ", ";
  st1 << "\nmodHE: ";
  for (const auto& it : pars->modHE)
    st1 << it << ", ";
  st1 << "\nmaxDepth: ";
  for (const auto& it : pars->maxDepth)
    st1 << it << ", ";
  st1 << "\nlayHB: ";
  for (const auto& it : pars->layHB)
    st1 << it << ", ";
  st1 << "\nlayHE: ";
  for (const auto& it : pars->layHE)
    st1 << it << ", ";
  st1 << "\netaMin: ";
  for (const auto& it : pars->etaMin)
    st1 << it << ", ";
  st1 << "\netaMax: ";
  for (const auto& it : pars->etaMax)
    st1 << it << ", ";
  st1 << "\nnoff: ";
  for (const auto& it : pars->noff)
    st1 << it << ", ";
  st1 << "\nHBShift: ";
  for (const auto& it : pars->HBShift)
    st1 << it << ", ";
  st1 << "\nHEShift: ";
  for (const auto& it : pars->HEShift)
    st1 << it << ", ";
  st1 << "\nHFShift: ";
  for (const auto& it : pars->HFShift)
    st1 << it << ", ";
  edm::LogVerbatim("HCalGeom") << st1.str();

  std::ostringstream st2;
  for (const auto& it : pars->layerGroupEtaSim) {
    st2 << "layerGroupEtaSim" << it.layer << ": ";
    for (const auto& iit : it.layerGroup) {
      st2 << iit << ", ";
    }
  }
  st2 << "\netagroup: ";
  for (const auto& it : pars->etagroup)
    st2 << it << ", ";
  st2 << "\nphigroup: ";
  for (const auto& it : pars->phigroup)
    st2 << it << ", ";
  for (const auto& it : pars->layerGroupEtaRec) {
    st2 << "\nlayerGroupEtaRec" << it.layer << ": ";
    for (const auto& iit : it.layerGroup) {
      st2 << iit << ", ";
    }
  }
  st2 << "\ndzVcal: " << pars->dzVcal << "\n(Topology|Trigger)Mode: " << std::hex << pars->topologyMode << std::dec;
  edm::LogVerbatim("HCalGeom") << st2.str();

  const auto& parS = iSetup.getData(simparToken_);
  const HcalSimulationParameters* parsim = &parS;
  if (parsim != nullptr) {
    std::ostringstream st3;
    st3 << "\nattenuationLength_: ";
    for (const auto& it : parsim->attenuationLength_)
      st3 << it << ", ";
    st3 << "\nlambdaLimits_: ";
    for (const auto& it : parsim->lambdaLimits_)
      st3 << it << ", ";
    st3 << "\nshortFiberLength_: ";
    for (const auto& it : parsim->shortFiberLength_)
      st3 << it << ", ";
    st3 << "\nlongFiberLength_: ";
    for (const auto& it : parsim->longFiberLength_)
      st3 << it << ", ";
    edm::LogVerbatim("HCalGeom") << st3.str();

    std::ostringstream st4;
    st4 << "\npmtRight_: ";
    for (const auto& it : parsim->pmtRight_)
      st4 << it << ", ";
    st4 << "\npmtFiberRight_: ";
    for (const auto& it : parsim->pmtFiberRight_)
      st4 << it << ", ";
    st4 << "\npmtLeft_: ";
    for (const auto& it : parsim->pmtLeft_)
      st4 << it << ", ";
    st4 << "\npmtFiberLeft_: ";
    for (const auto& it : parsim->pmtFiberLeft_)
      st4 << it << ", ";
    edm::LogVerbatim("HCalGeom") << st4.str();

    std::ostringstream st5;
    st5 << "\nhfLevels_: ";
    for (const auto& it : parsim->hfLevels_)
      st5 << it << ", ";
    st5 << "\nhfNames_: ";
    for (const auto& it : parsim->hfNames_)
      st5 << it << ", ";
    st5 << "\nhfFibreNames_: ";
    for (const auto& it : parsim->hfFibreNames_)
      st5 << it << ", ";
    st5 << "\nhfPMTNames_: ";
    for (const auto& it : parsim->hfPMTNames_)
      st5 << it << ", ";
    st5 << "\nhfFibreStraightNames_: ";
    for (const auto& it : parsim->hfFibreStraightNames_)
      st5 << it << ", ";
    st5 << "\nhfFibreConicalNames_: ";
    for (const auto& it : parsim->hfFibreConicalNames_)
      st5 << it << ", ";
    st5 << "\nhcalMaterialNames_: ";
    for (const auto& it : parsim->hcalMaterialNames_)
      st5 << it << ", ";
    edm::LogVerbatim("HCalGeom") << st5.str();
  }
}

DEFINE_FWK_MODULE(HcalParametersAnalyzer);
