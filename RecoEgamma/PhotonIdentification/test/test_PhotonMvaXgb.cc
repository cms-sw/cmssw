#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "RecoEgamma/PhotonIdentification/interface/PhotonXGBoostEstimator.h"

#include <memory>

#define INPUT_LEN 10
#define NTREE_LIMIT_B_V1 168
#define NTREE_LIMIT_E_V1 158

#define NTREE_LIMIT_B_V2 144
#define NTREE_LIMIT_E_V2 99

float const vars_in[INPUT_LEN][9] = {
    {134.303, 0.945981, 0.0264346, 0.012448, 0.0208734, 113.405, 1.7446, 0.00437808, 0.303464},
    {95.8896, 0.988677, 0.0217735, 0.0137696, 0.0441448, 90.7534, 1.85852, 0.0176929, 0},
    {10.2401, 1.1569, 0.00201483, 3.62996e-08, 4.7182e-08, 10.2401, 1.78352, 0.030019, 0.544686},
    {29.9392, 0.697065, 0.0081139, 0.00515725, 0.0200072, 18.7519, -0.330034, 0.069339, 0},
    {108.427, 0.911677, 0.0246062, 0.0105294, 0.0453685, 85.5906, -1.74928, 0.0195397, 1.00826},
    {19.6606, 1, 0.00818396, 0.00822772, 0.0219786, 19.6606, -2.39845, 0.0758766, 0},
    {66.2052, 0.784169, 0.00864794, 0.0141328, 0.0932173, 40.2147, -1.37391, 0.00421972, 0.662302},
    {8.74049, 0.519034, 0.00893926, 0.00879872, 0.0741009, 4.24432, -0.913888, 0.0324049, 2.66463},
    {231.613, 1.13042, 0.0213042, 0.0278477, 0.017684, 231.613, -2.61615, 0.0236956, 0},
    {70.3165, 0.987047, 0.00893917, 0.00897895, 0.00935749, 65.8019, -0.495195, 0.042801, 0.331989}};

const float mva_score_v1[INPUT_LEN] = {
    0.98634, 0.97501, 0.00179, 0.70818, 0.98374, 0.00153, 0.97103, 0.00009, 0.00626, 0.95222};

const float mva_score_v2[INPUT_LEN] = {
    0.98382, 0.94038, 0.59126, 0.91911, 0.98032, 0.18934, 0.99078, 0.60751, 0.00389, 0.96929};

TEST_CASE("RecoEgamma/PhotonIdentification testXGBPhoton", "[TestPhotonMvaXgb]") {
  SECTION("Test mva_compute v1") {
    auto mvaEstimatorB = std::make_unique<PhotonXGBoostEstimator>(
        edm::FileInPath("RecoEgamma/PhotonIdentification/data/XGBoost/Photon_NTL_168_Barrel_v1.bin"), NTREE_LIMIT_B_V1);
    auto mvaEstimatorE = std::make_unique<PhotonXGBoostEstimator>(
        edm::FileInPath("RecoEgamma/PhotonIdentification/data/XGBoost/Photon_NTL_158_Endcap_v1.bin"), NTREE_LIMIT_E_V1);

    for (unsigned int i = 0; i < INPUT_LEN; i++) {
      float xgbScore;
      const float *v = vars_in[i];
      float etaSC = v[6];
      if (std::abs(etaSC) < 1.5)
        xgbScore = mvaEstimatorB->computeMva(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
      else
        xgbScore = mvaEstimatorE->computeMva(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
      CHECK_THAT(xgbScore, Catch::Matchers::WithinAbs(mva_score_v1[i], 0.0001));
    }
  }

  SECTION("Test mva_compute v2") {
    auto mvaEstimatorB = std::make_unique<PhotonXGBoostEstimator>(
        edm::FileInPath("RecoEgamma/PhotonIdentification/data/XGBoost/Photon_NTL_144_Barrel_v2.bin"), NTREE_LIMIT_B_V2);
    auto mvaEstimatorE = std::make_unique<PhotonXGBoostEstimator>(
        edm::FileInPath("RecoEgamma/PhotonIdentification/data/XGBoost/Photon_NTL_99_Endcap_v2.bin"), NTREE_LIMIT_E_V2);

    for (unsigned int i = 0; i < INPUT_LEN; i++) {
      float xgbScore;
      const float *v = vars_in[i];
      float etaSC = v[6];
      if (std::abs(etaSC) < 1.5)
        xgbScore = mvaEstimatorB->computeMva(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
      else
        xgbScore = mvaEstimatorE->computeMva(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
      CHECK_THAT(xgbScore, Catch::Matchers::WithinAbs(mva_score_v2[i], 0.0001));
    }
  }
}
