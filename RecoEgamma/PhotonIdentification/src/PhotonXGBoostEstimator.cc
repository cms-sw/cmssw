#include "RecoEgamma/PhotonIdentification/interface/PhotonXGBoostEstimator.h"
#include <sstream>

PhotonXGBoostEstimator::PhotonXGBoostEstimator(const edm::FileInPath& weightsFile, int best_ntree_limit) {
  booster_ = std::make_unique<pat::XGBooster>(weightsFile.fullPath());
  booster_->addFeature("rawEnergy");
  booster_->addFeature("r9");
  booster_->addFeature("sigmaIEtaIEta");
  booster_->addFeature("etaWidth");
  booster_->addFeature("phiWidth");
  booster_->addFeature("s4");
  booster_->addFeature("eta");
  booster_->addFeature("hOvrE");
  booster_->addFeature("ecalPFIso");

  best_ntree_limit_ = best_ntree_limit;
}

PhotonXGBoostEstimator::~PhotonXGBoostEstimator() {}

float PhotonXGBoostEstimator::computeMva(float rawEnergyIn,
                                         float r9In,
                                         float sigmaIEtaIEtaIn,
                                         float etaWidthIn,
                                         float phiWidthIn,
                                         float s4In,
                                         float etaIn,
                                         float hOvrEIn,
                                         float ecalPFIsoIn) const {
  booster_->set("rawEnergy", rawEnergyIn);
  booster_->set("r9", r9In);
  booster_->set("sigmaIEtaIEta", sigmaIEtaIEtaIn);
  booster_->set("etaWidth", etaWidthIn);
  booster_->set("phiWidth", phiWidthIn);
  booster_->set("s4", s4In);
  booster_->set("eta", etaIn);
  booster_->set("hOvrE", hOvrEIn);
  booster_->set("ecalPFIso", ecalPFIsoIn);

  return booster_->predict(best_ntree_limit_);
}
