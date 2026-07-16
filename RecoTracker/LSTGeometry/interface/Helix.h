#ifndef RecoTracker_LSTGeometry_interface_Helix_h
#define RecoTracker_LSTGeometry_interface_Helix_h

// This header contains implementations so that the standalone part of LSTCore can use it without having to link some extra library.

#include <algorithm>
#include <array>
#include <cmath>

#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

  struct Helix {
    std::array<float, 3> center;
    float radius;
    float phi;
    float lambda;
    int charge;

    Helix(std::array<float, 3> center, float radius, float phi, float lambda, int charge)
        : center(center), radius(radius), phi(phi_mpi_pi(phi)), lambda(lambda), charge(charge) {}

    // Clarification : phi was derived assuming a negatively charged particle would start
    // at the first quadrant. However the way signs are set up in the get_track_point function
    // implies the particle actually starts out in the fourth quadrant, and phi is measured from
    // the y axis as opposed to x axis in the expression provided in this function. Hence I tucked
    // in an extra pi/2 to account for these effects
    Helix(float pt, float vx, float vy, float vz, float mx, float my, float mz, int charge) : charge(charge) {
      radius = pt / (kC * kB);

      float t = 2. * std::asin(std::sqrt((vx - mx) * (vx - mx) + (vy - my) * (vy - my)) / (2. * radius));
      phi = std::numbers::pi_v<float> / 2. + std::atan((vy - my) / (vx - mx)) +
            ((vy - my) / (vx - mx) < 0) * (std::numbers::pi_v<float>)+charge * t / 2. +
            (my - vy < 0) * (std::numbers::pi_v<float> / 2.) - (my - vy > 0) * (std::numbers::pi_v<float> / 2.);

      center[0] = vx + charge * radius * std::sin(phi);
      center[1] = vy - charge * radius * std::cos(phi);
      center[2] = vz;
      lambda = std::atan((mz - vz) / (radius * t));
    }

    Helix(float pt, float eta, float phi, float vx, float vy, float vz, float charge) : phi(phi), charge(charge) {
      // Radius based on pt
      radius = pt / (kC * kB);

      // reference point vector which for sim track is the vertex point
      float ref_vec_x = vx;
      float ref_vec_y = vy;
      float ref_vec_z = vz;

      // The reference to center vector
      float inward_radial_vec_x = charge * radius * std::sin(phi);
      float inward_radial_vec_y = charge * radius * -std::cos(phi);
      float inward_radial_vec_z = 0;

      // Center point
      center[0] = ref_vec_x + inward_radial_vec_x;
      center[1] = ref_vec_y + inward_radial_vec_y;
      center[2] = ref_vec_z + inward_radial_vec_z;

      // Lambda
      lambda = std::atan(std::sinh(eta));
    }

    std::array<float, 4> point(float t) const {
      float x = center[0] - charge * radius * std::sin(phi - charge * t);
      float y = center[1] + charge * radius * std::cos(phi - charge * t);
      float z = center[2] + radius * std::tan(lambda) * t;
      float r = std::sqrt(x * x + y * y);
      return {x, y, z, r};
    }

    float inferT(std::array<float, 3> const& point) const {
      return (point[2] - center[2]) / (radius * std::tan(lambda));
    }

    std::tuple<float, float, float, float> pointFromRadius(float target_r) const {
      float cx = center[0], cy = center[1];
      float center_r = std::sqrt(cx * cx + cy * cy);

      float sin_val = (cx * cx + cy * cy + radius * radius - target_r * target_r) / (2.f * charge * radius * center_r);
      sin_val = std::clamp(sin_val, -1.f, 1.f);

      float offset = std::atan2(-cy, cx);
      float alpha = std::asin(sin_val);

      // Two solutions: theta = alpha - offset  or  pi - alpha - offset
      // theta = phi - charge*t  =>  t = charge*(phi - theta)
      auto to_t = [this](float theta) {
        float t = charge * (phi - theta);
        const float two_pi = 2.f * std::numbers::pi_v<float>;
        t = std::fmod(t, two_pi);
        if (t < 0.f)
          t += two_pi;
        return t;
      };

      float t1 = to_t(alpha - offset);
      float t2 = to_t(std::numbers::pi_v<float> - alpha - offset);

      // Prefer the smallest t in [0, pi], matching the original search range
      bool t1_ok = t1 <= std::numbers::pi_v<float>;
      bool t2_ok = t2 <= std::numbers::pi_v<float>;
      float t = (t1_ok && t2_ok) ? std::min(t1, t2) : t1_ok ? t1 : t2_ok ? t2 : std::min(t1, t2);

      float x = center[0] - charge * radius * std::sin(phi - charge * t);
      float y = center[1] + charge * radius * std::cos(phi - charge * t);
      float z = center[2] + radius * std::tan(lambda) * t;
      float r = std::sqrt(x * x + y * y);

      return std::make_tuple(x, y, z, r);
    }

    std::tuple<float, float, float, float> pointFromZ(float target_z) const {
      float t = (target_z - center[2]) / (radius * std::tan(lambda));

      float x = center[0] - charge * radius * std::sin(phi - charge * t);
      float y = center[1] + charge * radius * std::cos(phi - charge * t);
      float z = center[2] + radius * std::tan(lambda) * t;
      float r = std::sqrt(x * x + y * y);

      return std::make_tuple(x, y, z, r);
    }

    float compareRadius(std::array<float, 3> const& pt) const {
      float t = inferT(pt);
      auto [x, y, z, r] = point(t);
      float point_r = std::sqrt(pt[0] * pt[0] + pt[1] * pt[1]);
      return (point_r - r);
    }

    float compareXY(std::array<float, 3> const& pt) const {
      float t = inferT(pt);
      auto [x, y, z, r] = point(t);
      float xy_dist = std::sqrt(std::pow(pt[0] - x, 2) + std::pow(pt[1] - y, 2));
      return xy_dist;
    }
  };

}  // namespace lstgeometry

#endif
