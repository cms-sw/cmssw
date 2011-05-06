#ifndef ThirdHitZPrediction_H
#define ThirdHitZPrediction_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"


class  ThirdHitZPrediction {
public:
  typedef PixelRecoRange<float> Range;
  ThirdHitZPrediction(const GlobalPoint& p1, float erroRPhi1, float errorZ1,
                      const GlobalPoint& p2, float erroRPhi2, float errorZ2, 
                      double curvature, double  nSigma = 3.);
  Range operator()(const GlobalPoint& p3, float erroRPhi3) const;
private:
  GlobalPoint thePoint1, thePoint2;
  double theErrorXY1, theErrorZ1, theErrorXY2, theErrorZ2;
  double theCurvature;
  double theNSigma;
};
#endif

