#include "RecoTracker/LSTGeometry/interface/Common.h"

namespace lstgeometry {

  float degToRad(float degrees) { return degrees * (std::numbers::pi_v<float> / 180); }

  float phi_mpi_pi(float phi) {
    while (phi >= std::numbers::pi_v<float>)
      phi -= 2 * std::numbers::pi_v<float>;
    while (phi < -std::numbers::pi_v<float>)
      phi += 2 * std::numbers::pi_v<float>;
    return phi;
  }

  float roundAngle(float angle, float tol) {
    const float pi = std::numbers::pi_v<float>;
    if (std::fabs(angle) < tol) {
      return 0.0;
    } else if (std::fabs(angle - pi / 2) < tol) {
      return pi / 2;
    } else if (std::fabs(angle + pi / 2) < tol) {
      return -pi / 2;
    } else if (std::fabs(angle - pi) < tol || std::fabs(angle + pi) < tol) {
      return -pi;
    }
    return angle;
  }

  float roundCoordinate(float coord, float tol) {
    if (std::fabs(coord) < tol) {
      return 0.0;
    }
    return coord;
  }

  std::pair<float, float> getEtaPhi(float x, float y, float z, float refphi) {
    float phi = phi_mpi_pi(std::atan2(y, x) - refphi);
    float eta = std::asinh(z / std::sqrt(x * x + y * y));
    return std::make_pair(eta, phi);
  }

}  // namespace lstgeometry
