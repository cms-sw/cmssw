#include "RecoEgamma/PhotonIdentification/interface/PhotonXGBoostEstimator.h"
#include <sstream>

PhotonXGBoostEstimator::PhotonXGBoostEstimator(const edm::FileInPath& weightsFile, int best_ntree_limit) {
  XGBoosterCreate(NULL, 0, &booster_);
  XGBoosterLoadModel(booster_, weightsFile.fullPath().c_str());
  best_ntree_limit_ = best_ntree_limit;

  std::stringstream config;
  config << "{\"training\": false, \"type\": 0, \"iteration_begin\": 0, \"iteration_end\": " << best_ntree_limit_
         << ", \"strict_shape\": false}";
  config_ = config.str();
}

PhotonXGBoostEstimator::~PhotonXGBoostEstimator() { XGBoosterFree(booster_); }

namespace {
  enum inputIndexes {
    rawEnergy = 0,      // 0
    r9 = 1,             // 1
    sigmaIEtaIEta = 2,  // 2
    etaWidth = 3,       // 3
    phiWidth = 4,       // 4
    s4 = 5,             // 5
    eta = 6,            // 6
    hOvrE = 7,          // 7
    ecalPFIso = 8,      // 8
  };
}  // namespace

float PhotonXGBoostEstimator::computeMva(float rawEnergyIn,
                                         float r9In,
                                         float sigmaIEtaIEtaIn,
                                         float etaWidthIn,
                                         float phiWidthIn,
                                         float s4In,
                                         float etaIn,
                                         float hOvrEIn,
                                         float ecalPFIsoIn) const {
  float var[9];
  var[rawEnergy] = rawEnergyIn;
  var[r9] = r9In;
  var[sigmaIEtaIEta] = sigmaIEtaIEtaIn;
  var[etaWidth] = etaWidthIn;
  var[phiWidth] = phiWidthIn;
  var[s4] = s4In;
  var[eta] = etaIn;
  var[hOvrE] = hOvrEIn;
  var[ecalPFIso] = ecalPFIsoIn;

  DMatrixHandle dmat;
  XGDMatrixCreateFromMat(var, 1, 9, -999.9f, &dmat);
  uint64_t const* out_shape;
  uint64_t out_dim;
  const float* out_result = NULL;
  XGBoosterPredictFromDMatrix(booster_, dmat, config_.c_str(), &out_shape, &out_dim, &out_result);
  float ret = out_result[0];
  XGDMatrixFree(dmat);
  return ret;
}
