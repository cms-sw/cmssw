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
                               float tolerance);

  double phi(double curvature, double radius) const;
  double angle(double curvature, double radius) const;

  Range operator()(Range curvature, double radius) const;

  Range curvature(double transverseIP) const;
  double curvature(const Basic2DVector<double> &thirdPoint) const;
  double transverseIP(const Basic2DVector<double> &thirdPoint) const;

  // like PixelRecoLineRZ, but makes use of the bending computation
  // from the circle fit to get an actual Helix propagation
  class HelixRZ {
    public:
      HelixRZ() : circle(0) {}
      HelixRZ(const ThirdHitPredictionFromCircle *circle,
              double z1, double z2, double curvature);

      double zAtR(double r) const;
      double rAtZ(double z) const;

      static double maxCurvature(const ThirdHitPredictionFromCircle *circle,
                                 double z1, double z2, double z3);

    private:
      const ThirdHitPredictionFromCircle *circle;
      double curvature, z1, seg, dzdu;
  };

private:
  friend class HelixRZ;

  double invCenterOnAxis(const Basic2DVector<double> &thirdPoint) const;

  Basic2DVector<double> p1, center, axis;
  double delta, delta2, theTolerance;
};

#endif
