#include <cmath>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/TPUtils.h"

namespace emtf::phase2::tp {

  // _______________________________________________________________________
  // radians <-> degrees
  float degToRad(float deg) {
    constexpr float factor = M_PI / 180.;
    return deg * factor;
  }

  float radToDeg(float rad) {
    constexpr float factor = 180. / M_PI;

    return rad * factor;
  }

  // _______________________________________________________________________
  // phi range: [-180..180] or [-pi..pi]
  float wrapPhiDeg(float deg) {
    float twopi = 360.;
    float recip = 1.0 / twopi;

    return deg - (std::round(deg * recip) * twopi);
  }

  float wrapPhiRad(float rad) {
    const float twopi = M_PI * 2.;
    const float recip = 1.0 / twopi;

    return rad - (std::round(rad * recip) * twopi);
  }

  // _______________________________________________________________________
  // theta
  float calcThetaRadFromEta(float eta) {
    float theta = std::atan2(1.0, std::sinh(eta));  // cot(theta) = sinh(eta)

    return theta;
  }

  float calcThetaDegFromEta(float eta) {
    float theta = radToDeg(calcThetaRadFromEta(eta));

    return theta;
  }

  float calcThetaRadFromInt(int theta_int) {
    float theta = degToRad(calcThetaDegFromInt(theta_int));

    return theta;
  }

  float calcThetaDegFromInt(int theta_int) {
    float theta = static_cast<float>(theta_int);

    theta = theta * (45.0 - 8.5) / 128. + 8.5;

    return theta;
  }

  int calcThetaInt(int endcap, float theta) {  // theta in deg [0..180], endcap [-1, +1]
    theta = (endcap == -1) ? (180. - theta) : theta;
    theta = (theta - 8.5) * 128. / (45.0 - 8.5);

    int theta_int = static_cast<int>(std::round(theta));

    theta_int = (theta_int <= 0) ? 1 : theta_int;  // protect against invalid value

    return theta_int;
  }

  // _______________________________________________________________________
  // phi
  float calcPhiGlobDegFromLoc(int sector, float loc) {  // loc in deg, sector [1..6]
    float glob = loc + 15. + (60. * (sector - 1));

    glob = (glob >= 180.) ? (glob - 360.) : glob;

    return glob;
  }

  float calcPhiGlobRadFromLoc(int sector, float loc) {  // loc in rad, sector [1..6]
    float glob = degToRad(calcPhiGlobDegFromLoc(sector, radToDeg(loc)));

    return glob;
  }

  float calcPhiLocDegFromInt(int phi_int) {
    float loc = static_cast<float>(phi_int);

    loc = (loc / 60.) - 22.;

    return loc;
  }

  float calcPhiLocRadFromInt(int phi_int) {
    float loc = degToRad(calcPhiLocDegFromInt(phi_int));

    return loc;
  }

  float calcPhiLocDegFromGlob(int sector, float glob) {  // glob in deg [-180..180], sector [1..6]
    glob = wrapPhiDeg(glob);

    float loc = glob - 15. - (60. * (sector - 1));

    return loc;
  }

  int calcPhiInt(int sector, float glob) {  // glob in deg [-180..180], sector [1..6]
    float loc = calcPhiLocDegFromGlob(sector, glob);

    loc = ((loc + 22.) < 0.) ? (loc + 360.) : loc;
    loc = (loc + 22.) * 60.;

    int phi_int = static_cast<int>(std::round(loc));

    return phi_int;
  }

}  // namespace emtf::phase2::tp
