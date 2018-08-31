#include "DQMOffline/Trigger/interface/FunctionDefs.h"

template<>
std::function<float(const reco::GsfElectron&)> hltdqm::getUnaryFuncExtraFloat<reco::GsfElectron>(const std::string& varName){
  std::function<float(const reco::GsfElectron&)> varFunc;
  if(varName=="scEta") varFunc = scEtaFunc<reco::GsfElectron>;
  else if(varName=="hOverE") varFunc = &reco::GsfElectron::hcalOverEcal;
  return varFunc;
}

template<>
std::function<float(const reco::Photon&)> hltdqm::getUnaryFuncExtraFloat<reco::Photon>(const std::string& varName){
  std::function<float(const reco::Photon&)> varFunc;
  if(varName=="scEta") varFunc = scEtaFunc<reco::Photon>;
  else if(varName=="hOverE") varFunc = &reco::Photon::hadTowOverEm;
  return varFunc;
}
