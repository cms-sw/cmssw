#ifndef ThirdHitPredictionFromInvParabola_H
#define ThirdHitPredictionFromInvParabola_H

/** Check phi compatibility of a hit with a track
    constrained by a hit pair and TrackingRegion kinematical constraints.
    The "inverse parabola method" is used. 
    M.Hansroul, H.Jeremie, D.Savard NIM A270 (1998) 490. 
 */

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include <array>

#include "FWCore/Utilities/interface/Visibility.h"

class TrackingRegion;
class OrderedHitPair;

// Function for testing ThirdHitPredictionFromInvParabola
namespace test {
  namespace PixelTriplets_InvPrbl_prec {
    int test();
  }
  namespace PixelTriplets_InvPrbl_t {
    int test();
  }
}  // namespace test

class ThirdHitPredictionFromInvParabola {
  // For tests
  friend int test::PixelTriplets_InvPrbl_prec::test();
  friend int test::PixelTriplets_InvPrbl_t::test();

public:
  using Scalar = double;
  typedef TkRotation2D<Scalar> Rotation;
  typedef PixelRecoRange<float> Range;
  typedef PixelRecoRange<Scalar> RangeD;
  typedef Basic2DVector<Scalar> Point2D;

  ThirdHitPredictionFromInvParabola() {}
  ThirdHitPredictionFromInvParabola(Scalar x1, Scalar y1, Scalar x2, Scalar y2, Scalar ip, Scalar curv, Scalar tolerance)
      : theTolerance(tolerance) {
    init(x1, y1, x2, y2, ip, std::abs(curv));
  }

  ThirdHitPredictionFromInvParabola(
      const GlobalPoint &P1, const GlobalPoint &P2, Scalar ip, Scalar curv, Scalar tolerance);

  //  inline Range operator()(Scalar radius, int charge) const { return rangeRPhiSlow(radius,charge,1); }
  inline Range operator()(Scalar radius, int charge) const { return rangeRPhi(radius, charge); }

  inline Range operator()(Scalar radius) const { return rangeRPhi(radius); }

  Range rangeRPhi(Scalar radius, int charge) const;  //  __attribute__ ((optimize(3, "fast-math")));

  Range rangeRPhi(Scalar radius) const;

  // Range rangeRPhiSlow(Scalar radius, int charge, int nIter=5) const;

  void init(const GlobalPoint &P1, const GlobalPoint &P2, Scalar ip, Scalar curv) {
    init(P1.x(), P1.y(), P2.x(), P2.y(), ip, curv);
  }
  void init(Scalar x1, Scalar y1, Scalar x2, Scalar y2, Scalar ip, Scalar curv);

private:
  inline Scalar coeffA(Scalar impactParameter) const;
  inline Scalar coeffB(Scalar impactParameter) const;
  inline Scalar predV(Scalar u, Scalar ip) const;
  inline Scalar ipFromCurvature(Scalar curvature, bool pos) const;

  Point2D transform(Point2D const &p) const { return theRotation.rotate(p) / p.mag2(); }

  Point2D transformBack(Point2D const &p) const { return theRotation.rotateBack(p) / p.mag2(); }

private:
  Rotation theRotation;
  Scalar u1u2, overDu, pv, dv, su;

  // formula is symmetric for (ip,pos) -> (-ip,neg)
  inline void findPointAtCurve(Scalar radius, Scalar ip, Scalar &u, Scalar &v) const;

  RangeD theIpRangePlus, theIpRangeMinus;
  Scalar theTolerance;
  bool emptyP, emptyM;
};

ThirdHitPredictionFromInvParabola::Scalar ThirdHitPredictionFromInvParabola::coeffA(Scalar impactParameter) const {
  auto c = -pv * overDu;
  return c - u1u2 * impactParameter;
}

ThirdHitPredictionFromInvParabola::Scalar ThirdHitPredictionFromInvParabola::coeffB(Scalar impactParameter) const {
  auto c = dv * overDu;
  return c - su * impactParameter;
}

ThirdHitPredictionFromInvParabola::Scalar ThirdHitPredictionFromInvParabola::ipFromCurvature(Scalar curvature,
                                                                                             bool pos) const {
  Scalar overU1u2 = 1. / u1u2;
  Scalar inInf = -pv * overDu * overU1u2;
  return (pos ? inInf : -inInf) - curvature * overU1u2 * 0.5;
}

ThirdHitPredictionFromInvParabola::Scalar ThirdHitPredictionFromInvParabola::predV(Scalar u, Scalar ip) const {
  auto c = -(coeffA(ip) - coeffB(ip * u) - ip * u * u);
  return c;
}

void ThirdHitPredictionFromInvParabola::findPointAtCurve(Scalar r, Scalar ip, Scalar &u, Scalar &v) const {
  //
  // assume u=(1-alpha^2/2)/r v=alpha/r
  // solve qudratic equation neglecting aplha^4 term
  //
  Scalar A = coeffA(ip);
  Scalar B = coeffB(ip);

  // Scalar overR = 1./r;
  Scalar ipOverR = ip / r;  // *overR;

  Scalar a = 0.5 * B + ipOverR;
  Scalar c = -B + A * r - ipOverR;

  Scalar delta = 1 - 4 * a * c;
  // Scalar sqrtdelta = (delta > 0) ? std::sqrt(delta) : 0.;
  Scalar sqrtdelta = std::sqrt(delta);
  //  Scalar alpha = (-1+sqrtdelta)/(2*a);
  Scalar alpha = (-2 * c) / (1 + sqrtdelta);

  v = alpha;               // *overR
  Scalar d2 = 1. - v * v;  // overR*overR - v*v
  // u = (d2 > 0) ? std::sqrt(d2) : 0.;
  u = std::sqrt(d2);

  // u,v not rotated! not multiplied by 1/r
}

#endif
