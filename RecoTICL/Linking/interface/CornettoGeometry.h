#ifndef RecoTICL_Linking_CornettoGeometry_h
#define RecoTICL_Linking_CornettoGeometry_h

// Geometry of the Cornetto pair test: the cone around the anchor axis, which points from the origin
// to the anchor barycenter, and the eta-phi search window that contains every anchor for which a
// given trackster can pass the cone.

#include <algorithm>
#include <cmath>
#include <optional>

#include "DataFormats/Math/interface/Vector3D.h"

namespace ticl::cornetto {

  struct Cone {
    float maxBackward;      // max separation [cm] upstream of the anchor along its axis
    float maxLongitudinal;  // max separation [cm] downstream of the anchor along its axis
    float radius0;          // max transverse distance [cm] at zero separation
    float slope;            // growth of the max transverse distance per cm of separation

    // Largest transverse distance the cone accepts.
    float maxRadius() const { return radius0 + slope * std::max(maxBackward, maxLongitudinal); }
  };

  // The separation s of other along the anchor axis when other is inside the cone of anchor, else no
  // value. anchorRad is |anchor|. Each test accepts only on a true comparison, so a NaN input fails.
  inline std::optional<float> separationInCone(const Cone& cone,
                                               const math::XYZVectorF& anchor,
                                               float anchorRad,
                                               const math::XYZVectorF& other) {
    if (!(anchorRad > 0.f))
      return std::nullopt;
    const auto axis = anchor / anchorRad;
    const auto d = other - anchor;
    const float s = d.Dot(axis);
    if (!(s >= -cone.maxBackward && s <= cone.maxLongitudinal))
      return std::nullopt;
    const float rT = cone.radius0 + cone.slope * std::abs(s);
    if (!(std::max(0.f, d.Mag2() - s * s) <= rT * rT))
      return std::nullopt;
    return s;
  }

  struct Window {
    float dEta;  // half-width in |eta|
    float dPhi;  // half-width in phi
  };

  // Half-widths around the direction of other that contain the direction of every anchor for which
  // other can pass the cone. The anchor axis passes through the origin, so the transverse distance is
  // |other| sin(alpha), with alpha the angle between the two directions; the cone radius bounds alpha.
  // Each half-width is at most cap, plus an absolute margin.
  inline Window searchWindow(const Cone& cone, float otherAbsEta, float otherRad, float cap) {
    constexpr float kRadiusMargin = 0.1f;      // [cm], covers the float cancellation in the cone test
    constexpr float kRelativeMargin = 1.01f;   // covers the float error of the angular bound
    constexpr float kAbsoluteMargin = 1.e-5f;  // covers the rounding of the window edges
    constexpr float kMinPolarAngle = 1.e-4f;   // keeps the sine of the lower polar angle positive
    const float alpha = std::asin(std::min(1.f, (cone.maxRadius() + kRadiusMargin) / otherRad));
    const float theta = 2.f * std::atan(std::exp(-otherAbsEta));
    const float sinLow = std::sin(std::max(theta - alpha, kMinPolarAngle));
    const float etaBound = kRelativeMargin * alpha / sinLow;
    const float phiBound =
        kRelativeMargin * 2.f * std::asin(std::min(1.f, std::sin(0.5f * alpha) / std::sqrt(std::sin(theta) * sinLow)));
    return {std::min(cap, etaBound) + kAbsoluteMargin, std::min(cap, phiBound) + kAbsoluteMargin};
  }

}  // namespace ticl::cornetto

#endif
