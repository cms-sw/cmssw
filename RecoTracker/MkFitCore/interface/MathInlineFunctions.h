#ifndef RecoTracker_MkFitCore_interface_MathInlineFunctions_h
#define RecoTracker_MkFitCore_interface_MathInlineFunctions_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/MathInlineCore.h"

#include <cmath>
#include <vdt/atan2.h>
#include <vdt/cos.h>
#include <vdt/log.h>
#include <vdt/sin.h>
#include <vdt/sincos.h>
#include <vdt/tan.h>

namespace mkfit {

  inline float squashPhiGeneral(float phi) {
    return phi - std::floor(0.5 * Const::InvPI * (phi + Const::PI)) * Const::TwoPI;
  }

  inline float squashPhiMinimal(float phi) {
    return phi >= Const::PI ? phi - Const::TwoPI : (phi < -Const::PI ? phi + Const::TwoPI : phi);
  }

  inline float getRad2(float x, float y) { return x * x + y * y; }

  inline float getInvRad2(float x, float y) { return 1.0f / (x * x + y * y); }

  inline float getPhi(float x, float y) { return vdt::fast_atan2f(y, x); }

  inline float getTheta(float r, float z) { return vdt::fast_atan2f(r, z); }

  inline float getEta(float r, float z) { return -1.0f * vdt::fast_logf(vdt::fast_tanf(getTheta(r, z) / 2.0f)); }

  inline float getEta(float theta) { return -1.0f * vdt::fast_logf(vdt::fast_tanf(theta / 2.0f)); }

  inline float getEta(float x, float y, float z) {
    const float theta = vdt::fast_atan2f(std::sqrt(x * x + y * y), z);
    return -1.0f * vdt::fast_logf(vdt::fast_tanf(theta / 2.0f));
  }

  inline float getHypot(float x, float y) { return std::sqrt(x * x + y * y); }

  inline float getRadErr2(float x, float y, float exx, float eyy, float exy) {
    return (x * x * exx + y * y * eyy + 2.0f * x * y * exy) / getRad2(x, y);
  }

  inline float getInvRadErr2(float x, float y, float exx, float eyy, float exy) {
    return (x * x * exx + y * y * eyy + 2.0f * x * y * exy) / cube(getRad2(x, y));
  }

  inline float getPhiErr2(float x, float y, float exx, float eyy, float exy) {
    const float rad2 = getRad2(x, y);
    return (y * y * exx + x * x * eyy - 2.0f * x * y * exy) / (rad2 * rad2);
  }

  inline float getThetaErr2(
      float x, float y, float z, float exx, float eyy, float ezz, float exy, float exz, float eyz) {
    const float rad2 = getRad2(x, y);
    const float rad = std::sqrt(rad2);
    const float hypot2 = rad2 + z * z;
    const float dthetadx = x * z / (rad * hypot2);
    const float dthetady = y * z / (rad * hypot2);
    const float dthetadz = -rad / hypot2;
    return dthetadx * dthetadx * exx + dthetady * dthetady * eyy + dthetadz * dthetadz * ezz +
           2.0f * dthetadx * dthetady * exy + 2.0f * dthetadx * dthetadz * exz + 2.0f * dthetady * dthetadz * eyz;
  }

  inline float getEtaErr2(float x, float y, float z, float exx, float eyy, float ezz, float exy, float exz, float eyz) {
    const float rad2 = getRad2(x, y);
    const float detadx = -x / (rad2 * std::sqrt(1 + rad2 / (z * z)));
    const float detady = -y / (rad2 * std::sqrt(1 + rad2 / (z * z)));
    const float detadz = 1.0f / (z * std::sqrt(1 + rad2 / (z * z)));
    return detadx * detadx * exx + detady * detady * eyy + detadz * detadz * ezz + 2.0f * detadx * detady * exy +
           2.0f * detadx * detadz * exz + 2.0f * detady * detadz * eyz;
  }

  inline float getPxPxErr2(float ipt, float phi, float vipt, float vphi) {  // ipt = 1/pT, v = variance
    const float iipt2 = 1.0f / (ipt * ipt);                                 //iipt = 1/(1/pT) = pT
    float cosP;
    float sinP;
    vdt::fast_sincosf(phi, sinP, cosP);
    return iipt2 * (iipt2 * cosP * cosP * vipt + sinP * sinP * vphi);
  }

  inline float getPyPyErr2(float ipt, float phi, float vipt, float vphi) {  // ipt = 1/pT, v = variance
    const float iipt2 = 1.0f / (ipt * ipt);                                 //iipt = 1/(1/pT) = pT
    float cosP;
    float sinP;
    vdt::fast_sincosf(phi, sinP, cosP);
    return iipt2 * (iipt2 * sinP * sinP * vipt + cosP * cosP * vphi);
  }

  inline float getPzPzErr2(float ipt, float theta, float vipt, float vtheta) {  // ipt = 1/pT, v = variance
    const float iipt2 = 1.0f / (ipt * ipt);                                     //iipt = 1/(1/pT) = pT
    const float cotT = 1.0f / vdt::fast_tanf(theta);
    const float cscT = 1.0f / vdt::fast_sinf(theta);
    return iipt2 * (iipt2 * cotT * cotT * vipt + cscT * cscT * cscT * cscT * vtheta);
  }

}  // namespace mkfit

#endif  // RecoTracker_MkFitCore_interface_MathInlineFunctions_h
