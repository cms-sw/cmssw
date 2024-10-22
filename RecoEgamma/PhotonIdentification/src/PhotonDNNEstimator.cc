#include "RecoEgamma/PhotonIdentification/interface/PhotonDNNEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/FileInPath.h"

#include <iostream>
#include <fstream>
#include <memory>

using namespace std::placeholders;

inline uint photonModelSelector(const std::map<std::string, float>& vars, float etaThr) {
  /* 
  Selection of the model to be applied on the photon based on eta limit
  */
  const auto absEta = std::abs(vars.at("eta"));
  if (absEta <= etaThr) {
    return 0;
  } else {
    return 1;
  }
}

PhotonDNNEstimator::PhotonDNNEstimator(const egammaTools::DNNConfiguration& cfg, const bool useEBModelInGap)
    : dnnHelper_(cfg,
                 std::bind(photonModelSelector,
                           _1,
                           (useEBModelInGap) ? PhotonDNNEstimator::ecalBarrelMaxEtaWithGap
                                             : PhotonDNNEstimator::ecalBarrelMaxEtaNoGap),
                 PhotonDNNEstimator::dnnAvaibleInputs),
      useEBModelInGap_(useEBModelInGap) {}

std::vector<tensorflow::Session*> PhotonDNNEstimator::getSessions() const { return dnnHelper_.getSessions(); };

const std::vector<std::string> PhotonDNNEstimator::dnnAvaibleInputs = {{"pt",
                                                                        "eta",
                                                                        "hadTowOverEm",
                                                                        "trkSumPtHollowConeDR03",
                                                                        "EcalRecHit",
                                                                        "SigmaIetaIeta",
                                                                        "SigmaIetaIetaFull5x5",
                                                                        "SigmaIEtaIPhiFull5x5",
                                                                        "EcalPFClusterIso",
                                                                        "HcalPFClusterIso",
                                                                        "HasPixelSeed",
                                                                        "R9Full5x5",
                                                                        "hcalTower"}};

std::map<std::string, float> PhotonDNNEstimator::getInputsVars(const reco::Photon& photon) const {
  // Prepare a map with all the defined variables
  std::map<std::string, float> variables;
  variables["pt"] = photon.pt();
  variables["eta"] = photon.eta();
  variables["hadTowOverEm"] = photon.hadTowOverEmValid() ? photon.hadTowOverEm() : 0;
  variables["trkSumPtHollowConeDR03"] = photon.trkSumPtHollowConeDR03();
  variables["EcalRecHit"] = photon.ecalRecHitSumEtConeDR03();
  variables["SigmaIetaIeta"] = photon.sigmaIetaIeta();
  variables["SigmaIetaIetaFull5x5"] = photon.full5x5_sigmaIetaIeta();
  variables["SigmaIEtaIPhiFull5x5"] = photon.full5x5_showerShapeVariables().sigmaIetaIphi;
  variables["EcalPFClusterIso"] = photon.ecalPFClusterIso();
  variables["HcalPFClusterIso"] = photon.hcalPFClusterIso();
  variables["HasPixelSeed"] = (Int_t)photon.hasPixelSeed();
  variables["R9Full5x5"] = photon.full5x5_r9();
  variables["hcalTower"] = photon.hcalTowerSumEtConeDR03();
  variables["R9Full5x5"] = photon.full5x5_r9();
  // Define more variables here and use them directly in the model config!
  return variables;
}

std::vector<std::pair<uint, std::vector<float>>> PhotonDNNEstimator::evaluate(
    const reco::PhotonCollection& photons, const std::vector<tensorflow::Session*>& sessions) const {
  // Collect the map of variables for each candidate and call the dnnHelper
  // Scaling, model selection and running is performed in the helper
  std::vector<std::map<std::string, float>> inputs;
  for (const auto& photon : photons) {
    inputs.push_back(getInputsVars(photon));
  }
  return dnnHelper_.evaluate(inputs, sessions);
}
