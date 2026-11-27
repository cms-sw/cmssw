#include <cmath>
#include <random>

#include <catch2/catch_all.hpp>

#include "DataFormats/Math/interface/deltaPhi.h"
#include "RecoTICL/Linking/interface/CornettoGeometry.h"

using ticl::cornetto::Cone;

namespace {
  math::XYZVectorF fromEtaPhiZ(float eta, float phi, float z) {
    const float rho = std::abs(z) / std::sinh(std::abs(eta));
    return {rho * std::cos(phi), rho * std::sin(phi), z};
  }

  // Counts the compatible pairs and the pairs among them outside the search window of the child.
  std::pair<long, long> countMisses(const Cone& cone, float cap, long nPairs) {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> uEta(1.3f, 3.4f);
    std::uniform_real_distribution<float> uPhi(-3.14159265f, 3.14159265f);
    std::uniform_real_distribution<float> uZ(320.f, 520.f);
    std::uniform_real_distribution<float> uS(-cone.maxBackward, cone.maxLongitudinal);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    long inside = 0;
    long misses = 0;
    for (long k = 0; k < nPairs; ++k) {
      const float sign = u01(gen) < 0.5f ? -1.f : 1.f;
      const auto anchor = fromEtaPhiZ(uEta(gen), uPhi(gen), sign * uZ(gen));
      const float anchorRad = std::sqrt(anchor.Mag2());
      const auto axis = anchor / anchorRad;
      // A unit vector orthogonal to the axis, at a random azimuth around it.
      auto perp = axis.Cross(math::XYZVectorF(0.f, 0.f, 1.f));
      perp /= std::sqrt(perp.Mag2());
      const auto perp2 = axis.Cross(perp);
      const float psi = uPhi(gen);
      const auto dir = perp * std::cos(psi) + perp2 * std::sin(psi);
      const float s = uS(gen);
      const float rT = cone.radius0 + cone.slope * std::abs(s);
      // One third of the children sit on the cone edge, the others anywhere up to twice the radius.
      const float dT = (k % 3 == 0) ? rT : 2.f * rT * u01(gen);
      const auto child = anchor + axis * s + dir * dT;
      // A compatible pair passes the cone and the eta-phi cap, as in the plugin pair test.
      const float dEta = std::abs(std::abs(anchor.eta()) - std::abs(child.eta()));
      const float dPhi = std::abs(reco::deltaPhi(anchor.phi(), child.phi()));
      if (!ticl::cornetto::separationInCone(cone, anchor, anchorRad, child) || dEta > cap || dPhi > cap)
        continue;
      ++inside;
      const auto window = ticl::cornetto::searchWindow(cone, std::abs(child.eta()), std::sqrt(child.Mag2()), cap);
      if (!(dEta <= window.dEta && dPhi <= window.dPhi))
        ++misses;
    }
    return {inside, misses};
  }
}  // namespace

TEST_CASE("The search window contains every anchor of a compatible pair", "[CornettoGeometry]") {
  constexpr float kCap = 3.f;
  constexpr long kPairs = 2000000;
  for (const Cone cone : {Cone{20.f, 60.f, 4.f, 0.05f}, Cone{0.f, 60.f, 0.f, 0.f}, Cone{20.f, 120.f, 10.f, 0.2f}}) {
    const auto [inside, misses] = countMisses(cone, kCap, kPairs);
    INFO("cone " << cone.maxBackward << " " << cone.maxLongitudinal << " " << cone.radius0 << " " << cone.slope);
    REQUIRE(inside > kPairs / 10);
    REQUIRE(misses == 0);
  }
}

TEST_CASE("A NaN position fails the cone test", "[CornettoGeometry]") {
  const Cone cone{20.f, 60.f, 4.f, 0.05f};
  const math::XYZVectorF anchor(10.f, 10.f, 350.f);
  const math::XYZVectorF nan(std::nanf(""), 0.f, 350.f);
  REQUIRE_FALSE(ticl::cornetto::separationInCone(cone, anchor, std::sqrt(anchor.Mag2()), nan));
  REQUIRE_FALSE(ticl::cornetto::separationInCone(cone, nan, std::sqrt(nan.Mag2()), anchor));
}

TEST_CASE("The cone test is the same in the two endcaps", "[CornettoGeometry]") {
  const Cone cone{20.f, 60.f, 4.f, 0.05f};
  std::mt19937 gen(777);
  std::uniform_real_distribution<float> uEta(1.3f, 3.4f);
  std::uniform_real_distribution<float> uPhi(-3.14159265f, 3.14159265f);
  std::uniform_real_distribution<float> uZ(320.f, 520.f);
  std::uniform_real_distribution<float> uShift(-10.f, 10.f);
  auto mirror = [](const math::XYZVectorF& v) { return math::XYZVectorF(v.x(), v.y(), -v.z()); };
  for (int k = 0; k < 100000; ++k) {
    const auto anchor = fromEtaPhiZ(uEta(gen), uPhi(gen), uZ(gen));
    const auto other = anchor + math::XYZVectorF(uShift(gen), uShift(gen), 4.f * uShift(gen));
    const float rad = std::sqrt(anchor.Mag2());
    const auto s = ticl::cornetto::separationInCone(cone, anchor, rad, other);
    const auto sMirror = ticl::cornetto::separationInCone(cone, mirror(anchor), rad, mirror(other));
    REQUIRE(s.has_value() == sMirror.has_value());
    if (s)
      REQUIRE(*s == *sMirror);
  }
}
