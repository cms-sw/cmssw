#ifndef RecoTracker_LSTGeometry_interface_Common_h
#define RecoTracker_LSTGeometry_interface_Common_h

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>

namespace lstgeometry {

  constexpr float kB = 3.8;
  constexpr float kC = 0.00299792458;

  constexpr unsigned int kBarrelLayers = 6;
  constexpr unsigned int kEndcapLayers = 5;

  // For pixel maps
  constexpr unsigned int kNEta = 25;
  constexpr unsigned int kNPhi = 72;
  constexpr unsigned int kNZ = 25;
  constexpr std::array<float, 2> kPtBounds = {{2.0, 10'000.0}};
  // Range of the eta and z (pLS dz) binning; phi covers [-pi, pi]
  constexpr float kEtaMax = 2.6f;
  constexpr float kZMax = 30.f;

  // Superbin bins, shared by the pixel maps and the pLSs that look them up. The eta bin is not clamped, so that callers can
  // tell how far out of the range a value is; phi = pi and z >= kZMax go to the last bin.
  inline int etaBin(float eta) { return static_cast<int>(std::floor((eta + kEtaMax) * (kNEta / (2.f * kEtaMax)))); }
  inline int phiBin(float phi) {
    return std::clamp(static_cast<int>((phi + std::numbers::pi_v<float>)*(kNPhi / (2.f * std::numbers::pi_v<float>))),
                      0,
                      static_cast<int>(kNPhi) - 1);
  }
  inline int zBin(float z) {
    return std::min(static_cast<int>((std::clamp(z, -kZMax, kZMax) + kZMax) / (2.f * kZMax / kNZ)),
                    static_cast<int>(kNZ) - 1);
  }

  // This is defined as a constant in case the legacy value (123456789) needs to be used
  constexpr float kDefaultSlope = std::numeric_limits<float>::infinity();

  float degToRad(float degrees);
  float phi_mpi_pi(float phi);
  float roundAngle(float angle, float tol = 1e-3);
  float roundCoordinate(float coord, float tol = 1e-3);
  std::pair<float, float> getEtaPhi(float x, float y, float z, float refphi = 0);
}  // namespace lstgeometry

namespace lst {
  inline std::string floatToStr(float num, unsigned int precision = 1) {
    std::ostringstream outSS;
    outSS << std::setprecision(precision) << num;
    return outSS.str();
  }
}  // namespace lst

#endif
