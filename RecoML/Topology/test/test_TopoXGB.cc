#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <cmath>
#include <memory>
#include <vector>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "PhysicsTools/XGBoost/interface/XGBooster.h"

#define INPUT_LEN 5
#define N_MUONS 1
#define N_FEATURES (2 + 4 * N_MUONS)  // PFHT, MaxPNetB, mu_pt, mu_tkIso, mu_ecalIso, mu_hcalIso

// Each row: [pfht, maxPNetB, mu0_pt, mu0_tkIso, mu0_ecalIso, mu0_hcalIso]
// These are representative HLT-like values
const float vars_in[INPUT_LEN][N_FEATURES] = {
    {63.2921f, 0.0080455f, 40.7565f, 0.f, -1.12096f, -5.34428f},
    {281.672f, 0.120594f, 25.701f, 0.f, 0.235229f, -8.96128f},
    {106.037f, 0.144644f, 28.3136f, 0.f, -1.10701f, -11.4606f},
    {72.1984f, 0.0683462f, 28.3781f, 0.f, -0.467795f, -4.98114f},
};

// Expected scores — fill these in after running the model once on the inputs
// above and verifying the output is physically reasonable
const float expected_scores[INPUT_LEN] = {
    0.00278761f,
    0.9951f,
    0.709743f,
    0.13873f,

};

// Tolerance for score comparison
constexpr float kScoreTolerance = 1e-4f;

// Helper: build and configure a booster the same way the producer does
std::unique_ptr<pat::XGBooster> makeBooster(const std::string& modelPath) {
  auto booster = std::make_unique<pat::XGBooster>(modelPath);
  booster->addFeature("pfht");
  booster->addFeature("maxPNetB");
  for (unsigned int imu = 0; imu < N_MUONS; ++imu) {
    booster->addFeature("muon" + std::to_string(imu) + "_pt");
    booster->addFeature("muon" + std::to_string(imu) + "_tkIso");
    booster->addFeature("muon" + std::to_string(imu) + "_ecalIso");
    booster->addFeature("muon" + std::to_string(imu) + "_hcalIso");
  }
  return booster;
}

TEST_CASE("TopoMuonHtPNetBXGBProducer BDT score", "[TopoMuonBDT]") {
  const std::string modelPath = edm::FileInPath(
                                    "HLTrigger/HLTfilters/data/"
                                    "HLT_xgb_model_HH2b2W1L_1mu_HLTHT_Mupt-absiso_PNetB.json")
                                    .fullPath();

  SECTION("Scores match expected values") {
    auto booster = makeBooster(modelPath);

    for (unsigned int i = 0; i < INPUT_LEN; ++i) {
      std::vector<float> features(vars_in[i], vars_in[i] + N_FEATURES);
      const float score = booster->predict(features, 0);
      INFO("Input index " << i << ": score=" << score << " expected=" << expected_scores[i]);
      CHECK_THAT(score, Catch::Matchers::WithinAbs(expected_scores[i], kScoreTolerance));
    }
  }

  SECTION("Score is in valid range [0, 1]") {
    auto booster = makeBooster(modelPath);

    for (unsigned int i = 0; i < INPUT_LEN; ++i) {
      std::vector<float> features(vars_in[i], vars_in[i] + N_FEATURES);
      const float score = booster->predict(features, 0);
      CHECK(score >= 0.f);
      CHECK(score <= 1.f);
    }
  }

  SECTION("Zero-padded missing muon gives valid score") {
    auto booster = makeBooster(modelPath);

    // Simulate an event where the muon slots are zero-padded (no muon found)
    std::vector<float> features(N_FEATURES, 0.f);
    features[0] = 300.f;  // pfht
    features[1] = 0.6f;   // maxPNetB

    const float score = booster->predict(features, 0);
    CHECK(score >= 0.f);
    CHECK(score <= 1.f);
  }
}
