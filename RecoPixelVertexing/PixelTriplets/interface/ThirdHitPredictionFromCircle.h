#ifndef ThirdHitPredictionFromCircle_H
#define ThirdHitPredictionFromCircle_H

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"


class ThirdHitPredictionFromCircle {

public:
  typedef PixelRecoRange<float> Range;

  ThirdHitPredictionFromCircle(const GlobalPoint & P1, const GlobalPoint & P2,
                               double curv, double tolerance);

  Range operator()(double radius) const;
  Range transverseIP() const { return Range(radius - center1.r(), center2.r() - radius); }
  double curvature(const Basic2DVector<double> &thirdPoint) const;
  double transverseIP(const Basic2DVector<double> &thirdPoint) const;

private:
  class PointRPhi {
    public:
      PointRPhi() {}
      PointRPhi(double r, double phi) : theR(r), thePhi(phi) {}
      PointRPhi(const Basic2DVector<double> &p) : theR(p.mag()), thePhi(p.phi()) {}
      double r() const { return theR; }
      double phi() const { return thePhi; }

    private:
      double theR, thePhi;
  };

  double invCenterOnAxis(const Basic2DVector<double> &thirdPoint) const;

  float radius, theTolerance;
  PointRPhi center1, center2;
  Basic2DVector<double> p1, center, axis;
};

#endif
