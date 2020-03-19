#include "ThirdHitZPrediction.h"

namespace {
  template <class T>
  T sqr(T t) {
    return t * t;
  }
}  // namespace

ThirdHitZPrediction::Range ThirdHitZPrediction::operator()(const GlobalPoint& thePoint3, float erroRPhi3) const {
  double dR23 = (thePoint3 - thePoint2).perp();

  double slope = dR23 / dR12;
  if ((theCurvature > 1.e-4) && (std::abs(0.5 * dR23 * theCurvature) < 1.))
    slope = std::asin(0.5 * dR23 * theCurvature) / dS12;

  double z3 = thePoint2.z() + dZ12 * slope;

  double sqr_errorXY23 = sqr_errorXY2 + sqr(erroRPhi3);
  double error = sqrt(sqr((1 + dR23 / dR12) * theErrorZ2) + sqr(dR23 / dR12 * theErrorZ1) +
                      sqr(dZ12 / dR12) * sqr_errorXY23 + sqr((dZ12 / dR12) * (dR23 / dR12)) * sqr_errorXY12);
  error *= theNSigma;
  return Range(z3 - error, z3 + error);
}
