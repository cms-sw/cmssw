#ifndef RecoTracker_LSTGeometry_interface_Common_h
#define RecoTracker_LSTGeometry_interface_Common_h

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
