#include <cmath>
#include <random>
#include <vector>

#include <catch2/catch_all.hpp>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "RecoTICL/Linking/interface/CornettoFeatures.h"

using ticl::Trackster;
using ticl::cornetto::emProbability;

namespace {
  using Type = Trackster::ParticleType;
  const std::vector<int> kPhotonElectron{static_cast<int>(Type::photon), static_cast<int>(Type::electron)};
  const std::vector<int> kPhotonElectronPi0{
      static_cast<int>(Type::photon), static_cast<int>(Type::electron), static_cast<int>(Type::neutral_pion)};

  Trackster withProbabilities(std::initializer_list<std::pair<Type, float>> values) {
    Trackster ts;
    for (auto const& [type, p] : values)
      ts.setIdProbability(type, p);
    return ts;
  }
}  // namespace

TEST_CASE("A pi0-dominated trackster is EM with the neutral pion in the categories, not without",
          "[CornettoFeatures]") {
  const auto pi0 = withProbabilities({{Type::photon, 0.10f},
                                      {Type::electron, 0.05f},
                                      {Type::neutral_pion, 0.70f},
                                      {Type::charged_hadron, 0.10f},
                                      {Type::neutral_hadron, 0.05f}});
  REQUIRE_THAT(emProbability(pi0, kPhotonElectronPi0), Catch::Matchers::WithinAbs(0.85, 1e-6));
  REQUIRE_THAT(emProbability(pi0, kPhotonElectron), Catch::Matchers::WithinAbs(0.15, 1e-6));
  // The same thresholds as the type veto: 0.85 is above 0.6, 0.15 is below 0.4.
  REQUIRE(emProbability(pi0, kPhotonElectronPi0) > 0.6f);
  REQUIRE(emProbability(pi0, kPhotonElectron) < 0.4f);
}

TEST_CASE("The EM probability is normalised to the sum of the probabilities", "[CornettoFeatures]") {
  // Probabilities that do not sum to one: the EM fraction is the ratio.
  const auto ts = withProbabilities({{Type::photon, 0.3f}, {Type::charged_hadron, 0.3f}});
  REQUIRE_THAT(emProbability(ts, kPhotonElectron), Catch::Matchers::WithinAbs(0.5, 1e-6));
  REQUIRE_THAT(emProbability(ts, {static_cast<int>(Type::charged_hadron)}), Catch::Matchers::WithinAbs(0.5, 1e-6));
}

TEST_CASE("A trackster without a PID has EM probability -1", "[CornettoFeatures]") {
  REQUIRE(emProbability(Trackster{}, kPhotonElectronPi0) == -1.f);
  // A merged trackster carries only unknown.
  const auto merged = withProbabilities({{Type::unknown, 1.f}});
  REQUIRE(emProbability(merged, kPhotonElectronPi0) == -1.f);
}

TEST_CASE("The group energy is the member sum; its pt direction and EM probability are energy-weighted",
          "[CornettoFeatures]") {
  using ticl::cornetto::Features;
  using ticl::cornetto::groupFeatures;
  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  for (int trial = 0; trial < 100; ++trial) {
    const unsigned int n = 1 + trial % 40;
    Features f;
    f.resize(n);
    for (unsigned int i = 0; i < n; ++i) {
      // Barycenters in one endcap: |eta| in about [1.5, 3], z about 320 to 500 cm.
      const float z = 320.f + 180.f * u01(gen);
      const float eta = 1.5f + 1.5f * u01(gen);
      const float phi = 6.2831853f * (u01(gen) - 0.5f);
      const float r = z / std::sinh(eta);
      f.bary[i] = math::XYZVectorF(r * std::cos(phi), r * std::sin(phi), z);
      f.energy[i] = 0.1f + 50.f * u01(gen);
      f.eta[i] = f.bary[i].eta();
      f.phi[i] = f.bary[i].phi();
      f.rad[i] = std::sqrt(f.bary[i].Mag2());
      f.em[i] = u01(gen) < 0.3f ? -1.f : u01(gen);  // 30% without a PID
      f.time[i] = 0.f;
      f.timeError[i] = -1.f;
      f.valid[i] = true;
    }
    // A random partition into groups of consecutive members.
    std::vector<std::vector<unsigned int>> groups;
    for (unsigned int i = 0; i < n; ++i) {
      if (groups.empty() || u01(gen) < 0.4f)
        groups.emplace_back();
      groups.back().push_back(i);
    }
    const Features g = groupFeatures(f, groups);
    REQUIRE(g.energy.size() == groups.size());
    for (std::size_t k = 0; k < groups.size(); ++k) {
      // The reference: the raw energy of the merged trackster, the pt along the energy-weighted member barycenter
      // and the energy-weighted EM probability of the members with a PID.
      float energy = 0.f;
      float emWeight = 0.f;
      float em = 0.f;
      math::XYZVectorF weighted;
      for (unsigned int i : groups[k]) {
        energy += f.energy[i];
        weighted += f.bary[i] * f.energy[i];
        if (f.em[i] >= 0.f) {
          emWeight += f.energy[i];
          em += f.em[i] * f.energy[i];
        }
      }
      const float pt = energy / std::cosh(weighted.eta());
      const float emRef = emWeight > 0.f ? em / emWeight : -1.f;
      REQUIRE(g.valid[k]);
      REQUIRE(g.energy[k] == energy);
      REQUIRE_THAT(g.energy[k] / std::cosh(g.eta[k]), Catch::Matchers::WithinRel(pt, 1e-5f));
      REQUIRE_THAT(g.em[k], Catch::Matchers::WithinRel(emRef, 1e-6f));
    }
  }
}
