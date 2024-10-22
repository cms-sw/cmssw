/**GEMStripTopology
 * based on CSCRadialStripTopology and TrapezoidalStripTopology
 * \author Hyunyong Kim - TAMU
 */
#include "Geometry/CommonTopologies/interface/GEMStripTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <cmath>
#include <algorithm>

GEMStripTopology::GEMStripTopology(int ns, float aw, float dh, float r0)
    : numberOfStrips_(ns), angularWidth_(aw), detHeight_(dh), centreToIntersection_(r0) {
  assert(angularWidth_ != 0);
  assert(detHeight_ != 0);
  yAxisOrientation_ = 1;
  phiOfOneEdge_ = -(0.5 * numberOfStrips_) * angularWidth_ * yAxisOrientation_;
  yCentre_ = 0;
  LogTrace("GEMStripTopology") << "Constructing GEMStripTopology with"
                               << " nstrips = " << ns << " angular width = " << aw << " det. height = " << dh
                               << " r0 = " << r0 << "\n";
}

GEMStripTopology::GEMStripTopology(int ns, float aw, float dh, float r0, float yAx)
    : numberOfStrips_(ns), angularWidth_(aw), detHeight_(dh), centreToIntersection_(r0), yAxisOrientation_(yAx) {
  assert(angularWidth_ != 0);
  assert(detHeight_ != 0);
  phiOfOneEdge_ = -(0.5 * numberOfStrips_) * angularWidth_ * yAxisOrientation_;
  yCentre_ = 0;
  LogTrace("GEMStripTopology") << "Constructing GEMStripTopology with"
                               << " nstrips = " << ns << " angular width = " << aw << " det. height = " << dh
                               << " r0 = " << r0 << " yAxOrientation = " << yAx << "\n";
}

LocalPoint GEMStripTopology::localPosition(float strip) const {
  return LocalPoint(yAxisOrientation() * originToIntersection() * tan(stripAngle(strip)), 0);
}

LocalPoint GEMStripTopology::localPosition(const MeasurementPoint& mp) const {
  const float  // y = (L/cos(phi))*mp.y()*cos(phi)
      y(mp.y() * detHeight() + yCentreOfStripPlane()),
      x(yAxisOrientation() * yDistanceToIntersection(y) * std::tan(stripAngle(mp.x())));
  return LocalPoint(x, y);
}

LocalError GEMStripTopology::localError(float strip, float stripErr2) const {
  const double phi(stripAngle(strip)), t1(std::tan(phi)), t2(t1 * t1),
      tt(stripErr2 * std::pow(centreToIntersection() * angularWidth(), 2)),  // tangential sigma^2   *c2
      rr(std::pow(detHeight(), 2) * (1.f / 12.f)),  // radial sigma^2( uniform prob density along strip)  *c2

      xx(tt + t2 * rr), yy(t2 * tt + rr), xy(t1 * (rr - tt));

  return LocalError(xx, xy, yy);
}

LocalError GEMStripTopology::localError(const MeasurementPoint& mp, const MeasurementError& me) const {
  const double phi(stripAngle(mp.x())), s1(std::sin(phi)), c1(std::cos(phi));
  assert(c1 != 0);
  const double cs(s1 * c1), s2(s1 * s1),
      c2(1 - s2),  // rotation matrix

      T(angularWidth() * (centreToIntersection() + yAxisOrientation() * mp.y() * detHeight()) /
        c1),                // tangential measurement unit (local pitch)
      R(detHeight() / c1),  // radial measurement unit (strip length)
      tt(me.uu() * T * T),  // tangential sigma^2
      rr(me.vv() * R * R),  // radial sigma^2
      tr(me.uv() * T * R),

      xx(c2 * tt + 2 * cs * tr + s2 * rr), yy(s2 * tt - 2 * cs * tr + c2 * rr), xy(cs * (rr - tt) + tr * (c2 - s2));

  return LocalError(xx, xy, yy);
}

float GEMStripTopology::strip(const LocalPoint& lp) const {
  const float  // phi is measured from y axis --> sign of angle is sign of x * yAxisOrientation --> use atan2(x,y), not atan2(y,x)
      phi(std::atan2(lp.x(), yDistanceToIntersection(lp.y()))),
      aStrip((phi - yAxisOrientation() * phiOfOneEdge()) / angularWidth());
  return std::max(float(0), std::min((float)nstrips(), aStrip));
}

int GEMStripTopology::nearestStrip(const LocalPoint& lp) const {
  return std::min(nstrips(), static_cast<int>(std::max(float(0), strip(lp))) + 1);
}

MeasurementPoint GEMStripTopology::measurementPosition(const LocalPoint& lp) const {
  const float  // phi is [pi/2 - conventional local phi], use atan2(x,y) rather than atan2(y,x)
      phi(yAxisOrientation() * std::atan2(lp.x(), yDistanceToIntersection(lp.y())));
  return MeasurementPoint(yAxisOrientation() * (phi - phiOfOneEdge()) / angularWidth(),
                          (lp.y() - yCentreOfStripPlane()) / detHeight());
}

MeasurementError GEMStripTopology::measurementError(const LocalPoint& p, const LocalError& e) const {
  const double yHitToInter(yDistanceToIntersection(p.y()));
  assert(yHitToInter != 0);
  const double t(yAxisOrientation() * p.x() / yHitToInter),  // tan(strip angle)
      cs(t / (1 + t * t)), s2(t * cs), c2(1 - s2),           // rotation matrix

      T2(1. / (std::pow(angularWidth(), 2) *
               (std::pow(p.x(), 2) + std::pow(yHitToInter, 2)))),  // 1./tangential measurement unit (local pitch) ^2
      R2(c2 / std::pow(detHeight(), 2)),                           // 1./ radial measurement unit (strip length) ^2

      uu((c2 * e.xx() - 2 * cs * e.xy() + s2 * e.yy()) * T2), vv((s2 * e.xx() + 2 * cs * e.xy() + c2 * e.yy()) * R2),
      uv((cs * (e.xx() - e.yy()) + e.xy() * (c2 - s2)) * std::sqrt(T2 * R2));

  return MeasurementError(uu, uv, vv);
}

int GEMStripTopology::channel(const LocalPoint& lp) const { return std::min(int(strip(lp)), numberOfStrips_ - 1); }

float GEMStripTopology::pitch() const { return localPitch(LocalPoint(0, 0)); }

float GEMStripTopology::localPitch(const LocalPoint& lp) const {
  const int istrip = std::min(nstrips(), static_cast<int>(strip(lp)) + 1);  // which strip number
  const float fangle = stripAngle(static_cast<float>(istrip) - 0.5);        // angle of strip centre
  assert(std::cos(fangle - 0.5f * angularWidth()) != 0);
  return yDistanceToIntersection(lp.y()) * std::sin(angularWidth()) /
         std::pow(std::cos(fangle - 0.5f * angularWidth()), 2);
}

float GEMStripTopology::stripAngle(float strip) const {
  return phiOfOneEdge() + yAxisOrientation() * strip * angularWidth();
}

float GEMStripTopology::localStripLength(const LocalPoint& lp) const {
  assert(yDistanceToIntersection(lp.y()) != 0);
  return detHeight() * std::sqrt(1.f + std::pow(lp.x() / yDistanceToIntersection(lp.y()), 2));
}

float GEMStripTopology::yDistanceToIntersection(float y) const {
  return yAxisOrientation() * y + originToIntersection();
}

float GEMStripTopology::xOfStrip(int strip, float y) const {
  return yAxisOrientation() * yDistanceToIntersection(y) * std::tan(stripAngle(static_cast<float>(strip) - 0.5));
}
