#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

#include <catch2/catch_all.hpp>

#include "DataFormats/HGCalReco/interface/Common.h"
#include "RecoTICL/Linking/interface/CornettoLinking.h"

using ticl::cornetto::compatible;
using ticl::cornetto::Cone;
using ticl::cornetto::EmissionRule;
using ticl::cornetto::emits;
using ticl::cornetto::Features;
using ticl::cornetto::PairCuts;

namespace {
  constexpr float kPi = std::numbers::pi_v<float>;

  math::XYZVectorF fromEtaPhiZ(float absEta, float phi, float z) {
    const float rho = std::abs(z) / std::sinh(absEta);
    return {rho * std::cos(phi), rho * std::sin(phi), z};
  }

  void setNode(Features& f, unsigned int i, const math::XYZVectorF& bary, float energy, float em, float t, float tErr) {
    f.bary[i] = bary;
    f.energy[i] = energy;
    f.eta[i] = bary.eta();
    f.phi[i] = bary.phi();
    f.rad[i] = std::sqrt(bary.Mag2());
    f.em[i] = em;
    f.time[i] = t;
    f.timeError[i] = tErr;
    f.valid[i] = std::isfinite(bary.x()) && std::isfinite(bary.y()) && std::isfinite(bary.z()) && std::isfinite(energy);
  }

  // Clusters of tracksters around seeds: both endcaps, |eta| in [1, 3.8], a third of the seeds at the phi seam,
  // energies from a few values, and some invalid nodes.
  Features randomTracksters(std::mt19937& gen, unsigned int n) {
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    const std::vector<float> energies{1.f, 2.f, 5.f, 10.f, 20.f};
    const std::vector<float> ems{-1.f, 0.1f, 0.5f, 0.9f};
    Features f;
    f.resize(n);
    math::XYZVectorF seed;
    for (unsigned int i = 0; i < n; ++i) {
      if (i % 6 == 0) {
        const float absEta = 1.f + 2.8f * u01(gen);
        const float phi = u01(gen) < 0.33f ? kPi - 0.1f + 0.2f * u01(gen) : kPi * (2.f * u01(gen) - 1.f);
        const float z = (u01(gen) < 0.5f ? -1.f : 1.f) * (320.f + 200.f * u01(gen));
        seed = fromEtaPhiZ(absEta, phi, z);
      }
      const float seedRad = std::sqrt(seed.Mag2());
      const auto axis = seed / seedRad;
      auto perp = axis.Cross(math::XYZVectorF(0.f, 0.f, 1.f));
      perp /= std::sqrt(perp.Mag2());
      const auto perp2 = axis.Cross(perp);
      const float psi = 2.f * kPi * u01(gen);
      const float s = -30.f + 110.f * u01(gen);
      const float dT = 10.f * u01(gen);
      const auto bary = seed + axis * s + (perp * std::cos(psi) + perp2 * std::sin(psi)) * dT;
      const float energy = energies[static_cast<unsigned int>(u01(gen) * energies.size()) % energies.size()];
      const float em = ems[static_cast<unsigned int>(u01(gen) * ems.size()) % ems.size()];
      // Times local to the barycenter, with a spread of about the time error.
      const float t = s * ticl::cornetto::kInvSpeedOfLight + 0.1f * (u01(gen) - 0.5f);
      const float tErr = u01(gen) < 0.5f ? 0.03f : -1.f;
      setNode(f, i, bary, energy, em, t, tErr);
    }
    // Invalid nodes: a NaN barycenter, an infinite energy and a NaN energy.
    if (n > 10) {
      setNode(f, 3, math::XYZVectorF(std::nanf(""), 0.f, 400.f), 5.f, -1.f, 0.f, -1.f);
      setNode(f, 4, f.bary[0], std::numeric_limits<float>::infinity(), -1.f, 0.f, -1.f);
      setNode(f, 5, f.bary[0], std::nanf(""), -1.f, 0.f, -1.f);
    }
    return f;
  }

  // The parent search over all pairs.
  template <typename Allowed, typename Test>
  std::vector<unsigned int> bruteForceParents(const Features& f, const Allowed& allowed, const Test& test) {
    const auto n = static_cast<unsigned int>(f.energy.size());
    std::vector<unsigned int> parents(n);
    for (unsigned int i = 0; i < n; ++i) {
      unsigned int best = i;
      for (unsigned int j = 0; f.valid[i] && j < n; ++j) {
        if (j == i || !f.valid[j] || !ticl::cornetto::ranksAbove(f.energy, j, i))
          continue;
        if (best != i && !ticl::cornetto::ranksAbove(f.energy, j, best))
          continue;
        if (allowed(j, i) && test(j, i))
          best = j;
      }
      parents[i] = best;
    }
    return parents;
  }

  struct Counts {
    long parents = 0;
    long seam = 0;
    long zNegative = 0;
    long zPositive = 0;
    long equalEnergy = 0;
    long outsideTile = 0;
  };

  bool outsideTile(float eta) {
    return std::abs(eta) < ticl::TileConstants::minEta || std::abs(eta) > ticl::TileConstants::maxEta;
  }

  Counts compareWithBruteForce(const PairCuts& cuts, const Cone& cone, const Cone& emCone, bool restrictAllowed) {
    std::mt19937 gen(2468);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    Counts counts;
    for (int trial = 0; trial < 60; ++trial) {
      const unsigned int n = 30 + 5 * trial;
      const Features f = randomTracksters(gen, n);
      std::vector<bool> mask(n, true);
      if (restrictAllowed)
        for (unsigned int i = 0; i < n; ++i)
          mask[i] = u01(gen) < 0.7f;
      const auto allowed = [&](unsigned int a, unsigned int b) { return mask[a] && (mask[b] || b % 2 == 0); };
      const auto test = [&](unsigned int a, unsigned int b) { return compatible(cuts, cone, emCone, f, a, b); };
      auto tiles = std::make_unique<ticl::cornetto::EndcapTiles>();
      ticl::cornetto::fillTiles(f, *tiles);
      const auto parents = ticl::cornetto::findParents(f, *tiles, cone, cuts.etaWindow, allowed, test);
      const auto reference = bruteForceParents(f, allowed, test);
      REQUIRE(parents == reference);
      for (unsigned int i = 0; i < n; ++i) {
        const unsigned int p = reference[i];
        if (p == i)
          continue;
        ++counts.parents;
        counts.seam += std::abs(f.phi[i] - f.phi[p]) > kPi;
        counts.zNegative += f.bary[i].z() < 0.f;
        counts.zPositive += f.bary[i].z() > 0.f;
        counts.equalEnergy += f.energy[i] == f.energy[p];
        counts.outsideTile += outsideTile(f.eta[i]) || outsideTile(f.eta[p]);
      }
    }
    return counts;
  }

  void requireAllCases(const Counts& c) {
    INFO("parents " << c.parents << ", seam " << c.seam << ", z<0 " << c.zNegative << ", z>0 " << c.zPositive
                    << ", equal energy " << c.equalEnergy << ", outside the tile " << c.outsideTile);
    REQUIRE(c.parents > 1000);
    REQUIRE(c.seam > 0);
    REQUIRE(c.zNegative > 0);
    REQUIRE(c.zPositive > 0);
    REQUIRE(c.equalEnergy > 0);
    REQUIRE(c.outsideTile > 0);
  }

  const Cone kCone{20.f, 60.f, 4.f, 0.05f};
  const Cone kEMCone{20.f, 60.f, 1.5f, 0.05f};
}  // namespace

TEST_CASE("The tile parent search equals the search over all pairs", "[CornettoLinking]") {
  SECTION("Default cuts with the time gate and the type veto") {
    requireAllCases(compareWithBruteForce(PairCuts{0.3f, 0.6f, true, 9.f, 0.0025f}, kCone, kEMCone, false));
  }
  SECTION("Default cuts with a restricted allowed predicate") {
    requireAllCases(compareWithBruteForce(PairCuts{0.3f, 0.6f, true, 9.f, 0.0025f}, kCone, kEMCone, true));
  }
  SECTION("A wide cone and an eta window just below the limit, without time gate and type veto") {
    const float etaWindow = std::nextafter(kPi * (1.f - 2.f / ticl::TileConstants::nPhiBins), 0.f);
    const Cone wide{50.f, 200.f, 60.f, 0.5f};
    requireAllCases(compareWithBruteForce(PairCuts{etaWindow, 0.f, false, 0.f, 0.f}, wide, wide, false));
  }
}

TEST_CASE("The pair test", "[CornettoLinking]") {
  const PairCuts cuts{0.3f, 0.6f, true, 9.f, 0.0025f};
  Features f;
  f.resize(2);
  const auto anchor = fromEtaPhiZ(2.f, 1.f, 400.f);
  const float rad = std::sqrt(anchor.Mag2());
  const auto axis = anchor / rad;
  auto perp = axis.Cross(math::XYZVectorF(0.f, 0.f, 1.f));
  perp /= std::sqrt(perp.Mag2());
  // b is 10 cm downstream of a and 3 cm off its axis: inside the cone, outside the EM cone.
  const auto other = anchor + axis * 10.f + perp * 3.f;
  const float flight = 10.f * ticl::cornetto::kInvSpeedOfLight;
  setNode(f, 0, anchor, 10.f, 0.1f, 0.f, 0.03f);
  setNode(f, 1, other, 5.f, 0.1f, flight, 0.03f);
  REQUIRE(compatible(cuts, kCone, kEMCone, f, 0, 1));

  SECTION("An EM anchor uses the EM cone") {
    f.em[0] = 0.9f;
    f.em[1] = 0.9f;
    REQUIRE_FALSE(compatible(cuts, kCone, kEMCone, f, 0, 1));
  }
  SECTION("The type veto refuses an EM and a hadronic trackster, unless one has no PID or the veto is off") {
    f.em[1] = 0.9f;
    REQUIRE_FALSE(compatible(cuts, kCone, kEMCone, f, 0, 1));
    f.em[0] = -1.f;
    REQUIRE(compatible(cuts, kCone, kEMCone, f, 0, 1));
    f.em[0] = 0.1f;
    REQUIRE(compatible(PairCuts{0.3f, 0.f, true, 9.f, 0.0025f}, kCone, kCone, f, 0, 1));
  }
  SECTION("The time gate refuses a late trackster, unless an error is not positive or the gate is off") {
    f.time[1] = flight + 1.f;
    REQUIRE_FALSE(compatible(cuts, kCone, kEMCone, f, 0, 1));
    REQUIRE(compatible(PairCuts{0.3f, 0.6f, false, 9.f, 0.0025f}, kCone, kEMCone, f, 0, 1));
    f.timeError[1] = -1.f;
    REQUIRE(compatible(cuts, kCone, kEMCone, f, 0, 1));
  }
  SECTION("Tracksters in different endcaps are not compatible") {
    setNode(f, 1, math::XYZVectorF(other.x(), other.y(), -other.z()), 5.f, 0.1f, flight, 0.03f);
    REQUIRE_FALSE(compatible(cuts, kCone, kEMCone, f, 0, 1));
  }
  SECTION("The eta window refuses a pair inside the cone") {
    REQUIRE_FALSE(compatible(PairCuts{1e-4f, 0.6f, true, 9.f, 0.0025f}, kCone, kEMCone, f, 0, 1));
  }
}

TEST_CASE("The emission rule on the threshold edges", "[CornettoLinking]") {
  const EmissionRule rule{1.f, 0.5f, 2.f, 1.f, 0.f, 0.6f};
  const auto below = [](float x) { return std::nextafter(x, 0.f); };

  SECTION("Hadronic group: the energy and the pt thresholds are inclusive") {
    REQUIRE(emits(rule, 1.f, 0.5f, -1.f));
    REQUIRE_FALSE(emits(rule, below(1.f), 0.5f, -1.f));
    REQUIRE_FALSE(emits(rule, 1.f, below(0.5f), -1.f));
    REQUIRE_FALSE(emits(rule, std::nanf(""), 0.5f, -1.f));
    REQUIRE_FALSE(emits(rule, 1.f, std::nanf(""), -1.f));
  }
  SECTION("minEmittedEnergyAnyPt removes the pt threshold, the energy threshold still applies") {
    const EmissionRule anyPt{1.f, 0.5f, 2.f, 1.f, 5.f, 0.6f};
    REQUIRE(emits(anyPt, 5.f, 0.1f, -1.f));
    REQUIRE_FALSE(emits(anyPt, below(5.f), 0.1f, -1.f));
    const EmissionRule lowAnyPt{1.f, 0.5f, 2.f, 1.f, 0.5f, 0.6f};
    REQUIRE_FALSE(emits(lowAnyPt, below(1.f), 0.1f, -1.f));
    REQUIRE(emits(lowAnyPt, 1.f, 0.1f, -1.f));
  }
  SECTION("EM group: EM probability above typeVetoProbability uses the EM thresholds") {
    REQUIRE(emits(rule, 2.f, 1.f, 0.61f));
    REQUIRE_FALSE(emits(rule, below(2.f), 1.f, 0.61f));
    REQUIRE_FALSE(emits(rule, 2.f, below(1.f), 0.61f));
    // At the threshold the group is not EM.
    REQUIRE(emits(rule, 1.f, 0.5f, 0.6f));
    // typeVetoProbability 0: no group is EM.
    const EmissionRule noVeto{1.f, 0.5f, 2.f, 1.f, 0.f, 0.f};
    REQUIRE(emits(noVeto, 1.f, 0.5f, 0.9f));
  }
}
