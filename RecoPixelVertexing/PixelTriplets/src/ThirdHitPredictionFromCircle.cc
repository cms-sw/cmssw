#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "FWCore/Utilities/interface/Likely.h"

// there are tons of safety checks.
// Try to move all of the out the regular control flow using gcc magic

typedef Basic3DVector<double> Point3D;
typedef Basic2DVector<double> Point2D;

namespace {
  template<class T> static inline T sqr(T t) { return t * t; }
  template<class T> static inline T sgn(T t) { return std::signbit(t) ? -1. : 1.; }
  template<class T> static inline T clamped_acos(T t)
  { return unlikely(t <= -1) ? M_PI : unlikely(t >= 1) ? T(0) : std::acos(t); }
  template<class T> static inline T clamped_sqrt(T t)
  { return likely(t > 0) ? std::sqrt(t) : T(0); }
}

ThirdHitPredictionFromCircle::ThirdHitPredictionFromCircle( 
	const GlobalPoint& P1, const GlobalPoint& P2, float tolerance)
  : p1(P1.x(), P1.y()), theTolerance(tolerance)
{
  Point2D p2(P2.x(), P2.y());
  Point2D diff = 0.5 * (p2 - p1);
  delta2 = diff.mag2();
  delta = std::sqrt(delta2);
  axis = Point2D(-diff.y(), diff.x()) / delta;
  center = p1 + diff;
}

double ThirdHitPredictionFromCircle::phi(double curvature, double radius) const
{
  double phi;
  if (unlikely(std::abs(curvature) < 1.0e-5)) {
    double cos = (center * axis) / radius;
    phi = axis.phi() - clamped_acos(cos);
  } else {
    double sign = sgn(curvature);
    double radius2 = sqr(1.0 / curvature);
    double orthog = clamped_sqrt(radius2 - delta2);
    Basic2DVector<double> lcenter = center - sign * orthog * axis;
    double rc2 = lcenter.mag2();
    double cos = (rc2 + sqr(radius) - radius2) /
      (2. *std:: sqrt(rc2) * radius);
    phi = lcenter.phi() + sign * clamped_acos(cos);
 }

  while(unlikely(phi >= M_PI)) phi -= 2. * M_PI;
  while(unlikely(phi < -M_PI)) phi += 2. * M_PI;

  return phi;
}

double ThirdHitPredictionFromCircle::angle(double curvature, double radius) const
{
  if (unlikely(std::abs(curvature) < 1.0e-5)) {
    double sin = (center * axis) / radius;
    return sin / clamped_sqrt(1 - sqr(sin));
  } else {
    double radius2 = sqr(1.0 / curvature);
    double orthog = clamped_sqrt(radius2 - delta2);
    Basic2DVector<double> lcenter = center - sgn(curvature) * orthog * axis;
 
    double cos = (radius2 + sqr(radius) - lcenter.mag2()) *
                 curvature / (2. * radius);
    return - cos / clamped_sqrt(1 - sqr(cos));
 }
}

ThirdHitPredictionFromCircle::Range
ThirdHitPredictionFromCircle::operator()(Range curvature, double radius) const
{
  double phi1 = phi(curvature.second, radius);
  double phi2 = phi(curvature.first, radius);

  while(unlikely(phi2 <  phi1)) phi2 += 2. * M_PI; 

  return Range(phi1 * radius - theTolerance, phi2 * radius + theTolerance);
}

ThirdHitPredictionFromCircle::Range
ThirdHitPredictionFromCircle::curvature(double transverseIP) const
{
  // this is a mess.  Use a CAS and lots of drawings to verify...

  transverseIP = std::abs(transverseIP);
  double transverseIP2 = sqr(transverseIP);
  double tip = axis * center;
  double tip2 = sqr(tip);
  double lip = axis.x() * center.y() - axis.y() * center.x();
  double lip2 = sqr(lip);

  double origin = std::sqrt(tip2 + lip2);
  double tmp1 = lip2 + tip2 - transverseIP2;
  double tmp2 = 2. * (tip - transverseIP) * (tip + transverseIP);
  double tmp3 = 2. * delta * origin;
  double tmp4 = tmp1 + delta2;
  double tmp5 = 2. * delta * lip;

  // I am probably being overly careful here with border cases
  // but you never know what crap you might get fed

  double u1, u2;
  if (unlikely(tmp4 - tmp5 < 1.0e-5)) {
    u1 = -0.;	// yes, I am making use of signed zero
    u2 = +0.;	// -> no -ffast-math please
  } else {
    if (unlikely(std::abs(tmp2) < 1.0e-5)) {
      // the denominator is zero
      // this means that one of the tracks will be straight
      // and the other can be computed from the limit of the equation
      double tmp = lip2 - delta2;
      u1 = INFINITY;	// and require 1 / sqrt(inf^2 + x) = 0 (with x > 0)
      u2 = (sqr(0.5 * tmp) - delta2 * tip2) / (tmp * tip);
      if (tip < 0)
        std::swap(u1, u2);
    } else {
      double tmp6 = (tmp4 - tmp5) * (tmp4 + tmp5);
      if (unlikely(tmp6 < 1.0e-5)) {
        u1 = -0.;
        u2 = +0.;
      } else {
        double tmp7 = tmp6 > 0 ? (transverseIP * std::sqrt(tmp6) / tmp2) : 0.;
        double tmp8 = tip * (tmp1 - delta2) / tmp2;
        // the two quadratic solutions
        u1 = tmp8 + tmp7;
        u2 = tmp8 - tmp7;
      }
    }

    if (tmp4 <= std::abs(tmp3)) {
      if ((tmp3 < 0) == (tip < 0))
        u2 = +0.;
      else
        u1 = -0.;
    }
  }

  return Range(sgn(u1) / std::sqrt(sqr(u1) + delta2),
               sgn(u2) / std::sqrt(sqr(u2) + delta2));
}

double ThirdHitPredictionFromCircle::invCenterOnAxis(const Point2D &p2) const
{
  Point2D delta = p2 - p1;
  Point2D axis2 = Point2D(-delta.y(), delta.x()) / delta.mag();
  Point2D diff = p1 + 0.5 * delta - center;
  double a = diff.y() * axis2.x() - diff.x() * axis2.y();
  double b = axis.y() * axis2.x() - axis.x() * axis2.y();
  return b / a;
}

double ThirdHitPredictionFromCircle::curvature(const Point2D &p2) const
{
  double invDist = invCenterOnAxis(p2);
  double invDist2 = sqr(invDist);
  double curv = std::sqrt(invDist2 / (1. + invDist2 * delta2));
  return sgn(invDist) * curv;
}

double ThirdHitPredictionFromCircle::transverseIP(const Point2D &p2) const
{
  double invDist = invCenterOnAxis(p2);
  if (unlikely(std::abs(invDist) < 1.0e-5))
    return std::abs(p2 * axis);
  else {
    double dist = 1.0 / invDist;
    double radius = std::sqrt(sqr(dist) + delta2);
    return std::abs((center + axis * dist).mag() - radius);
  }
}

ThirdHitPredictionFromCircle::HelixRZ::HelixRZ(
  const ThirdHitPredictionFromCircle *circle, double z1, double z2, double curv) :
  circle(circle), curvature(curv), z1(z1)
{
  double absCurv = std::abs(curv);
  seg = circle->delta;

  if (likely(absCurv > 1.0e-5)) {
    seg *= absCurv;
    seg = seg < -1.0 ? -M_PI_2 : seg > 1.0 ? M_PI_2 : std::asin(seg);
    seg /= absCurv;
  }

  seg *= 2.;
  dzdu = likely(std::abs(seg) > 1.0e-5) ? ((z2 - z1) / seg) : 99999.0;
}

double ThirdHitPredictionFromCircle::HelixRZ::maxCurvature(
  const ThirdHitPredictionFromCircle *circle, double z1, double z2, double z3)
{
  static const double maxAngle = M_PI;
  double halfAngle = (0.5 * maxAngle) * (z2 - z1) / (z3 - z1);
  if (unlikely(halfAngle <= 0.0))
    return 0.0;

  return std::sin(halfAngle) / circle->delta;
}

double ThirdHitPredictionFromCircle::HelixRZ::zAtR(double r) const
{
  if (unlikely(std::abs(curvature) < 1.0e-5)) {
     double tip = circle->axis * circle->p1;
     double lip = circle->axis.y() * circle->p1.x() -
                  circle->axis.x() * circle->p1.y();
     return z1 + (std::sqrt(sqr(r) - sqr(tip)) - lip) * dzdu;
  }

  double radius = 1.0 / curvature;
  double radius2 = sqr(radius);
  double orthog = sgn(curvature) * clamped_sqrt(radius2 - circle->delta2);
  Point2D center = circle->center + orthog * circle->axis;

  double b2 = center.mag2();
  double b = std::sqrt(b2);

  double cos1 = 0.5 * (radius2 + b2 - sqr(r)) * curvature / b;
  double cos2 = 0.5 * (radius2 + b2 - circle->p1.mag2()) * curvature / b;

  double phi1 = clamped_acos(cos1);
  double phi2 = clamped_acos(cos2);

  // more plausbility checks needed...
  // the two circles can have two possible intersections
  double u1 = std::abs((phi1 - phi2) * radius);
  double u2 = std::abs((phi1 + phi2) * radius);

  return z1 + ((u1 >= seg && u1 < u2)? u1 : u2) * dzdu;
}

double ThirdHitPredictionFromCircle::HelixRZ::rAtZ(double z) const
{
  if (unlikely(std::abs(dzdu) < 1.0e-5))
    return 99999.0;

  if (unlikely(std::abs(curvature) < 1.0e-5)) {
    double tip = circle->axis * circle->p1;
    double lip = circle->axis.y() * circle->p1.x() -
                 circle->axis.x() * circle->p1.y();
    return std::sqrt(sqr(tip) + sqr(lip + (z - z1) / dzdu));
  }

  // we won't go below that (see comment below)
  double minR = (2. * circle->center - circle->p1).mag();

  double phi = curvature * (z - z1) / dzdu;

  if (unlikely(std::abs(phi) > 2. * M_PI)) {
    // with a helix we can get problems here - this is used to get the limits
    // however, if phi gets large, we get into the regions where we loop back
    // to smaller r's.  The question is - do we care about these tracks?
    // The answer is probably no:  Too much pain, and the rest of the
    // tracking won't handle looping tracks anyway.
    // So, what we do here is to return nothing smaller than the radius
    // than any of the two hits, i.e. the second hit, which is presumably
    // outside of the 1st hit.

    return minR;
  }

  double radius = 1. / curvature;
  double orthog = sgn(curvature) * clamped_sqrt(sqr(radius) - circle->delta2);
  Point2D center = circle->center + orthog * circle->axis;
  Point2D rel = circle->p1 - center;

  double c = cos(phi);
  double s = sin(phi);

  Point2D p(center.x() + c * rel.x() - s * rel.y(),
            center.y() + s * rel.x() + c * rel.y());

  return std::max(minR, p.mag());
}
