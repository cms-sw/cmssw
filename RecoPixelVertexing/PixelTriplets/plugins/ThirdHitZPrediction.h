#ifndef ThirdHitZPrediction_H
#define ThirdHitZPrediction_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include <cmath>

class ThirdHitZPrediction {
public:
  typedef PixelRecoRange<float> Range;
  ThirdHitZPrediction(const GlobalPoint& p1,
                      float erroRPhi1,
                      float errorZ1,
                      const GlobalPoint& p2,
                      float erroRPhi2,
                      float errorZ2,
                      double curvature,
                      double nSigma = 3.)
      : thePoint2(p2),
        sqr_errorXY12(erroRPhi1 * erroRPhi1 + erroRPhi2 * erroRPhi2),
        sqr_errorXY2(erroRPhi2 * erroRPhi2),
        theErrorZ1(errorZ1),
        theErrorZ2(errorZ2),
        theCurvature(curvature),
        theNSigma(nSigma) {
    auto d = p2 - p1;
    dR12 = d.perp();
    if (dR12 < 1.e-5)
      dR12 = 1.e-5;
    dS12 = std::abs(0.5 * dR12 * theCurvature) < 1 ? std::asin(0.5 * dR12 * theCurvature) : 1.;
    dZ12 = d.z();
  }
  Range operator()(const GlobalPoint& p3, float erroRPhi3) const;

private:
  GlobalPoint thePoint2;
  double dR12, dZ12, dS12;
  double sqr_errorXY12;
  double sqr_errorXY2;
  double theErrorZ1, theErrorZ2;
  double theCurvature;
  double theNSigma;
};
#endif
