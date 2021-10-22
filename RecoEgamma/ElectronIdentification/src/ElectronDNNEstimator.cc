#include "RecoEgamma/ElectronIdentification/interface/ElectronDNNEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/FileInPath.h"

#include <iostream>
#include <fstream>
#include <memory>

using namespace std::placeholders;

inline uint electronModelSelector(const std::map<std::string, float>& vars, float ptThr, float etaThr) {
  /* 
  Selection of the model to be applied on the electron based on pt/eta cuts or whatever selection
  */
  const auto pt = vars.at("pt");
  const auto absEta = std::abs(vars.at("eta"));
  if (pt < ptThr)
    return 0;
  else {
    if (absEta <= etaThr) {
      return 1;
    } else {
      return 2;
    }
  }
}

ElectronDNNEstimator::ElectronDNNEstimator(const egammaTools::DNNConfiguration& cfg, const bool useEBModelInGap)
    : dnnHelper_(cfg,
                 std::bind(electronModelSelector,
                           _1,
                           ElectronDNNEstimator::ptThreshold,
                           (useEBModelInGap) ? ElectronDNNEstimator::ecalBarrelMaxEtaWithGap
                                             : ElectronDNNEstimator::ecalBarrelMaxEtaNoGap),
                 ElectronDNNEstimator::dnnAvaibleInputs),
      useEBModelInGap_(useEBModelInGap) {}

std::vector<tensorflow::Session*> ElectronDNNEstimator::getSessions() const { return dnnHelper_.getSessions(); };

const std::vector<std::string> ElectronDNNEstimator::dnnAvaibleInputs = {{"pt",
                                                                          "eta",
                                                                          "fbrem",
                                                                          "abs(deltaEtaSuperClusterTrackAtVtx)",
                                                                          "abs(deltaPhiSuperClusterTrackAtVtx)",
                                                                          "full5x5_sigmaIetaIeta",
                                                                          "full5x5_hcalOverEcal",
                                                                          "eSuperClusterOverP",
                                                                          "full5x5_e1x5",
                                                                          "eEleClusterOverPout",
                                                                          "closestCtfTrackNormChi2",
                                                                          "closestCtfTrackNLayers",
                                                                          "gsfTrack.missing_inner_hits",
                                                                          "dr03TkSumPt",
                                                                          "dr03EcalRecHitSumEt",
                                                                          "dr03HcalTowerSumEt",
                                                                          "gsfTrack.normalizedChi2",
                                                                          "superCluster.eta",
                                                                          "ecalPFClusterIso",
                                                                          "hcalPFClusterIso",
                                                                          "numberOfBrems",
                                                                          "abs(deltaEtaSeedClusterTrackAtCalo)",
                                                                          "hadronicOverEm",
                                                                          "full5x5_e2x5Max",
                                                                          "full5x5_e5x5"}};

std::map<std::string, float> ElectronDNNEstimator::getInputsVars(const reco::GsfElectron& ele) const {
  // Prepare a map with all the defined variables
  std::map<std::string, float> variables;
  reco::TrackRef myTrackRef = ele.closestCtfTrackRef();
  bool validKF = (myTrackRef.isNonnull() && myTrackRef.isAvailable());
  variables["pt"] = ele.pt();
  variables["eta"] = ele.eta();
  variables["fbrem"] = ele.fbrem();
  variables["abs(deltaEtaSuperClusterTrackAtVtx)"] = std::abs(ele.deltaEtaSuperClusterTrackAtVtx());
  variables["abs(deltaPhiSuperClusterTrackAtVtx)"] = std::abs(ele.deltaPhiSuperClusterTrackAtVtx());
  variables["full5x5_sigmaIetaIeta"] = ele.full5x5_sigmaIetaIeta();
  variables["full5x5_hcalOverEcal"] = ele.full5x5_hcalOverEcalValid() ? ele.full5x5_hcalOverEcal() : 0;
  variables["eSuperClusterOverP"] = ele.eSuperClusterOverP();
  variables["full5x5_e1x5"] = ele.full5x5_e1x5();
  variables["eEleClusterOverPout"] = ele.eEleClusterOverPout();
  variables["closestCtfTrackNormChi2"] = ele.closestCtfTrackNormChi2();
  variables["closestCtfTrackNLayers"] = ele.closestCtfTrackNLayers();
  variables["gsfTrack.missing_inner_hits"] =
      (validKF) ? myTrackRef->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) : -1.;
  variables["dr03TkSumPt"] = ele.dr03TkSumPt();
  variables["dr03EcalRecHitSumEt"] = ele.dr03EcalRecHitSumEt();
  variables["dr03HcalTowerSumEt"] = ele.dr03HcalTowerSumEt();
  variables["gsfTrack.normalizedChi2"] = (validKF) ? myTrackRef->normalizedChi2() : 0;
  variables["superCluster.eta"] = ele.superCluster()->eta();
  variables["ecalPFClusterIso"] = ele.ecalPFClusterIso();
  variables["hcalPFClusterIso"] = ele.hcalPFClusterIso();
  variables["numberOfBrems"] = ele.numberOfBrems();
  variables["abs(deltaEtaSeedClusterTrackAtCalo)"] = std::abs(ele.deltaEtaSeedClusterTrackAtCalo());
  variables["hadronicOverEm"] = ele.hcalOverEcalValid() ? ele.hadronicOverEm() : 0;
  variables["full5x5_e2x5Max"] = ele.full5x5_e2x5Max();
  variables["full5x5_e5x5"] = ele.full5x5_e5x5();
  // Define more variables here and use them directly in the model config!
  return variables;
}

std::vector<std::vector<float>> ElectronDNNEstimator::evaluate(
    const reco::GsfElectronCollection& electrons, const std::vector<tensorflow::Session*>& sessions) const {
  // Collect the map of variables for each candidate and call the dnnHelper
  // Scaling, model selection and running is performed in the helper
  std::vector<std::map<std::string, float>> inputs;
  for (const auto& ele : electrons) {
    inputs.push_back(getInputsVars(ele));
  }
  return dnnHelper_.evaluate(inputs, sessions);
}
