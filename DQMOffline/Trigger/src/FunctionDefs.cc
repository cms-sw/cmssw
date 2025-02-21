#include "DQMOffline/Trigger/interface/FunctionDefs.h"

template <>
std::function<float(const reco::GsfElectron&)> hltdqm::getUnaryFuncExtraFloat<reco::GsfElectron>(
    const std::string& varName) {
  std::function<float(const reco::GsfElectron&)> varFunc;
  if (varName == "scEta")
    varFunc = scEtaFunc<reco::GsfElectron>;
  else if (varName == "hOverE")
    varFunc = [](const reco::GsfElectron& ele) -> float { return ele.hcalOverEcal(); };
  return varFunc;
}

template <>
std::function<float(const reco::Photon&)> hltdqm::getUnaryFuncExtraFloat<reco::Photon>(const std::string& varName) {
  std::function<float(const reco::Photon&)> varFunc;
  if (varName == "scEta")
    varFunc = scEtaFunc<reco::Photon>;
  else if (varName == "hOverE")
    varFunc = [](const reco::Photon& pho) -> float { return pho.hadTowOverEm(); };
  return varFunc;
}

template <>
std::function<float(const HLTGenValObject&)> hltdqm::getUnaryFuncExtraFloat<HLTGenValObject>(
    const std::string& varName) {
  std::function<float(const HLTGenValObject&)> varFunc;

  if (varName == "ptRes")
    varFunc = &HLTGenValObject::ptRes;
  else if (varName == "etaRes")
    varFunc = &HLTGenValObject::etaRes;
  else if (varName == "phiRes")
    varFunc = &HLTGenValObject::phiRes;
  else if (varName == "massRes")
    varFunc = &HLTGenValObject::massRes;
  return varFunc;
}