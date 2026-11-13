// Checks the photon transformer ID model and its preprocessing file (cms-data) with libtorch.
// Two synthetic photons (barrel with two PF candidates; endcap without PF candidates and with three preshower
// strips) are evaluated one at a time and in one padded batch, as in PhotonTransformerIDProducer, and compared with
// the outputs of the same model evaluated in python.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "PhysicsTools/PyTorch/interface/Model.h"

// libtorch defines a glog-style CHECK macro; use the Catch2 one in this test
#undef CHECK
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

namespace {

  const std::string kDirectory = "RecoEgamma/PhotonIdentification/data/PhotonTransformerID/Run3_2024_v1/";

  // python reference: sigmoid(logit) and the four features of each photon, evaluated alone
  const float kExpectedScore[2] = {0.0394388, 0.358999};
  const float kExpectedFeatures[2][4] = {{0.316854, -2.33651, -1.01425, 2.4435},
                                         {1.11427, 0.339961, 0.0339673, -0.763421}};
  constexpr float kTolerance = 1e-4;

  struct Photon {
    std::vector<float> image;  // 32 x 32, standardised
    std::vector<float> pf;     // 18 per candidate
    std::vector<float> es;     // 6 per strip
    std::vector<float> geom;   // eta, phi, is barrel
  };

  std::vector<float> standardisedImage(const std::map<std::pair<int, int>, float>& energies,
                                       const nlohmann::json& rechit) {
    std::vector<float> mean, sigma;
    for (const auto& row : rechit.at("cell_mean"))
      for (const auto& value : row)
        mean.push_back(value.get<float>());
    for (const auto& row : rechit.at("cell_std"))
      for (const auto& value : row)
        sigma.push_back(value.get<float>());
    const float eps = rechit.at("log_eps").get<float>();
    std::vector<float> image(32 * 32, 0.f);
    for (int cell = 0; cell < 32 * 32; ++cell) {
      const auto it = energies.find({cell / 32, cell % 32});
      const float energy = it == energies.end() ? 0.f : it->second;
      float value = std::log(std::max(0.f, energy) + eps);
      value = sigma[cell] > 0.f ? (value - mean[cell]) / sigma[cell] : 0.f;
      image[cell] = std::isfinite(value) ? value : 0.f;
    }
    return image;
  }

  // Evaluates the photons in one batch (padded and masked) and returns the scores and features.
  std::pair<std::vector<float>, std::vector<float>> evaluate(cms::torch::Model& model,
                                                             const std::vector<const Photon*>& photons) {
    const int n = photons.size();
    int maxPF = 0, maxES = 0;
    for (const auto* photon : photons) {
      maxPF = std::max<int>(maxPF, photon->pf.size() / 18);
      maxES = std::max<int>(maxES, photon->es.size() / 6);
    }
    auto image = torch::zeros({n, 1, 32, 32});
    auto aux = torch::zeros({n, 18});
    auto pf = torch::zeros({n, maxPF, 18});
    auto pfMask = torch::zeros({n, maxPF}, torch::kBool);
    auto es = torch::zeros({n, maxES, 6});
    auto esMask = torch::zeros({n, maxES}, torch::kBool);
    auto geom = torch::zeros({n, 3});
    for (int i = 0; i < n; ++i) {
      const Photon& photon = *photons[i];
      std::copy(photon.image.begin(), photon.image.end(), image[i].data_ptr<float>());
      std::copy(photon.pf.begin(), photon.pf.end(), pf[i].data_ptr<float>());
      pfMask[i].slice(0, 0, photon.pf.size() / 18).fill_(true);
      if (!photon.es.empty()) {
        std::copy(photon.es.begin(), photon.es.end(), es[i].data_ptr<float>());
        esMask[i].slice(0, 0, photon.es.size() / 6).fill_(true);
      }
      std::copy(photon.geom.begin(), photon.geom.end(), geom[i].data_ptr<float>());
    }
    std::vector<torch::IValue> inputs{image, aux, pf, pfMask, es, esMask, geom};
    const auto output = model.forward(inputs).toTuple();
    const auto scores = torch::sigmoid(output->elements()[0].toTensor()).reshape({-1}).contiguous();
    const auto features = output->elements()[1].toTensor().contiguous();
    REQUIRE(scores.numel() == n);
    REQUIRE(features.dim() == 2);
    REQUIRE(features.size(0) == n);
    REQUIRE(features.size(1) == 4);
    return {std::vector<float>(scores.data_ptr<float>(), scores.data_ptr<float>() + n),
            std::vector<float>(features.data_ptr<float>(), features.data_ptr<float>() + 4 * n)};
  }

}  // namespace

TEST_CASE("RecoEgamma/PhotonIdentification photon transformer ID model", "[PhotonTransformerID]") {
  // as PyTorchService in cmsRun; once per process (Catch2 re-enters the test case for each section)
  static const bool singleThreaded = [] {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    return true;
  }();
  REQUIRE(singleThreaded);

  std::ifstream file(edm::FileInPath(kDirectory + "preprocessing.json").fullPath());
  REQUIRE(file.is_open());
  const auto preprocessing = nlohmann::json::parse(file);
  REQUIRE(preprocessing.at("model").at("n_features").get<int>() == 4);
  REQUIRE(preprocessing.at("hlf").at("feat_mean").size() == 18);

  cms::torch::Model model(edm::FileInPath(kDirectory + "model.pt").fullPath());

  // barrel photon with two PF candidates and no preshower strips
  Photon barrel;
  barrel.image = standardisedImage(
      {{{16, 16}, 30.f}, {{16, 17}, 8.f}, {{15, 16}, 5.f}, {{17, 16}, 4.f}, {{16, 15}, 3.f}, {{15, 17}, 1.5f}},
      preprocessing.at("rechit"));
  barrel.pf = {0.5f,  0.1f, -0.2f, 0.4f,  0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, -1.f, -1.2f, 0.3f,  -0.4f, 0.6f,
               -0.8f, 0.2f, 0.1f,  -0.5f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, -2.f, -2.1f, -0.6f, 0.9f,  1.1f};
  barrel.geom = {0.5f, 1.f, 1.f};

  // endcap photon without PF candidates (one all-zero candidate) and with three preshower strips
  Photon endcap;
  endcap.image = standardisedImage({{{16, 16}, 40.f}, {{16, 17}, 10.f}, {{17, 16}, 6.f}, {{15, 15}, 2.f}},
                                   preprocessing.at("rechit"));
  endcap.pf = std::vector<float>(18, 0.f);
  endcap.es = {0.01f, -0.02f, 1.2f, 1.f, 0.f, 1.f, 0.05f, 0.f, 0.5f, 0.f, 1.f, 1.f, -0.3f, 0.4f, -0.5f, 1.f, 0.f, 0.f};
  endcap.geom = {2.f, 0.f, 0.f};

  const std::vector<const Photon*> photons{&barrel, &endcap};

  SECTION("one photon at a time") {
    for (int i = 0; i < 2; ++i) {
      const auto [scores, features] = evaluate(model, {photons[i]});
      CHECK_THAT(scores[0], Catch::Matchers::WithinAbs(kExpectedScore[i], kTolerance));
      for (int k = 0; k < 4; ++k)
        CHECK_THAT(features[k], Catch::Matchers::WithinAbs(kExpectedFeatures[i][k], kTolerance));
    }
  }

  SECTION("both photons in one padded batch") {
    const auto [scores, features] = evaluate(model, photons);
    for (int i = 0; i < 2; ++i) {
      CHECK_THAT(scores[i], Catch::Matchers::WithinAbs(kExpectedScore[i], kTolerance));
      for (int k = 0; k < 4; ++k)
        CHECK_THAT(features[4 * i + k], Catch::Matchers::WithinAbs(kExpectedFeatures[i][k], kTolerance));
    }
  }
}
