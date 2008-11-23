#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"

typedef Basic3DVector<double> Point3D;
typedef Basic2DVector<double> Point2D;
typedef PixelRecoRange<double> Ranged;

namespace {
	template<class T> static inline T sqr(T t) { return t * t; }
}

ThirdHitPredictionFromCircle::ThirdHitPredictionFromCircle( 
    const GlobalPoint& P1, const GlobalPoint& P2, double curv, double tolerance)
  : radius(1./curv), theTolerance(tolerance), p1(P1.x(), P1.y())
{
  Point2D p2(P2.x(), P2.y());
  Point2D delta = p2 - p1;
  double dist2 = delta.mag2();
  double dist = std::sqrt(dist2);
  double orthog = std::sqrt(sqr(radius) - 0.25 * dist2);
  axis = Point2D(-delta.y(), delta.x()) / dist;
  Point2D scaledAxis = orthog * axis;
  center = p1 + 0.5 * delta;
  center1 = PointRPhi(center - scaledAxis);
  center2 = PointRPhi(center + scaledAxis);
}

ThirdHitPredictionFromCircle::Range ThirdHitPredictionFromCircle::operator()(
    double radius) const
{
  double cos1 = (sqr(center1.r()) + sqr(radius) - sqr(this->radius)) / (2. * center1.r() * radius);
  double cos2 = (sqr(center2.r()) + sqr(radius) - sqr(this->radius)) / (2. * center2.r() * radius);

  double phi1 = cos1 <= -1.0 ? M_PI : cos1 >= 1.0 ? 0.0 : std::acos(cos1);
  double phi2 = cos2 <= -1.0 ? M_PI : cos2 >= 1.0 ? 0.0 : std::acos(cos2);

  phi1 = center1.phi() + phi1;
  phi2 = center2.phi() - phi2;

  while(phi1 >= M_PI) phi1 -= 2. * M_PI;
  while(phi1 < -M_PI) phi1 += 2. * M_PI;
  while(phi2 >= M_PI) phi2 -= 2. * M_PI;
  while(phi2 < phi1) phi2 += 2. * M_PI;

  return Range(phi1 * radius - theTolerance, phi2 * radius + theTolerance);
}

double ThirdHitPredictionFromCircle::invCenterOnAxis(const Point2D &p2) const
{
  Point2D delta = p2 - p1;
  double dist2 = delta.mag2();
  Point2D axis2 = Point2D(-delta.y(), delta.x()) / std::sqrt(dist2);
  Point2D diff = p1 + 0.5 * delta - center;
  double a = diff.y() * axis2.x() - diff.x() * axis2.y();
  double b = axis.y() * axis2.x() - axis.x() * axis2.y();
  return b / a;
}

double ThirdHitPredictionFromCircle::curvature(const Point2D &p2) const
{
  double invDist = invCenterOnAxis(p2);
  double invDist2 = sqr(invDist);
  double curv = std::sqrt(invDist2 / (1. + invDist2 * (center - p1).mag2()));
  return invDist < 0 ? -curv : curv;
}

double ThirdHitPredictionFromCircle::transverseIP(const Point2D &p2) const
{
  double invDist = invCenterOnAxis(p2);
  if (std::abs(invDist) < 1.0e-5)
    return std::abs(p2 * axis);
  else {
    double dist = 1.0 / invDist;
    double radius = std::sqrt(sqr(dist) + (center - p1).mag2());
    return std::abs((center + axis * dist).mag() - radius);
  }
}
