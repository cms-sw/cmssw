#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <cmath>
#include <algorithm>

//#define EDM_ML_DEBUG

TrapezoidalStripTopology::TrapezoidalStripTopology(int ns, float p, float l, float r0)
    : theNumberOfStrips(ns), thePitch(p), theDistToBeam(r0), theDetHeight(l) {
  theOffset = -theNumberOfStrips / 2. * thePitch;
  theYAxOr = 1;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CommonTopologies") << "Constructing TrapezoidalStripTopology with nstrips = " << ns << " pitch = " << p << " length = " << l << " r0 =" << r0;
#endif
}

TrapezoidalStripTopology::TrapezoidalStripTopology(int ns, float p, float l, float r0, int yAx)
    : theNumberOfStrips(ns), thePitch(p), theDistToBeam(r0), theDetHeight(l), theYAxOr(yAx) {
  theOffset = -theNumberOfStrips / 2. * thePitch;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CommonTopologies") << "Constructing TrapezoidalStripTopology with nstrips = " << ns << " pitch = " << p << " length = " << l << " r0 =" << r0 << " yAxOrientation =" << yAx;
#endif
}

LocalPoint TrapezoidalStripTopology::localPosition(float strip) const {
  return LocalPoint(strip * thePitch + theOffset, 0.0);
}

LocalPoint TrapezoidalStripTopology::localPosition(const MeasurementPoint& mp) const {
  float y = mp.y() * theDetHeight;
  float x = (mp.x() * thePitch + theOffset) * (theYAxOr * y + theDistToBeam) / theDistToBeam;
  return LocalPoint(x, y);
}

LocalError TrapezoidalStripTopology::localError(float strip, float stripErr2) const {
  float lt, lc2, ls2, lslc;
  float localL2, localP2;
  float sl2, sp2;
  // angle from strip to local frame (see CMS TN / 95-170)
  lt = -(strip * thePitch + theOffset) * theYAxOr / theDistToBeam;
  lc2 = 1.f / (1. + lt * lt);
  lslc = lt * lc2;
  ls2 = 1.f - lc2;
  localL2 = theDetHeight * theDetHeight / lc2;
  localP2 = thePitch * thePitch * lc2;
  sl2 = localL2 / 12.;
  sp2 = stripErr2 * localP2;
  return LocalError(lc2 * sp2 + ls2 * sl2, lslc * (sp2 - sl2), ls2 * sp2 + lc2 * sl2);
}

LocalError TrapezoidalStripTopology::localError(const MeasurementPoint& mp, const MeasurementError& merr) const {
  float lt, lc2, ls2, lslc;
  float localL, localP;
  float sl2, sp2, spl;
  // angle from strip to local frame (see CMS TN / 95-170)
  lt = -(mp.x() * thePitch + theOffset) * theYAxOr / theDistToBeam;
  lc2 = 1. / (1. + lt * lt);
  lslc = lt * lc2;
  ls2 = 1.f - lc2;
  localL = theDetHeight / std::sqrt(lc2);
  localP = localPitch(localPosition(mp));
  sp2 = merr.uu() * localP * localP;
  sl2 = merr.vv() * localL * localL;
  spl = merr.uv() * localP * localL;
  return LocalError(lc2 * sp2 + ls2 * sl2 - 2 * lslc * spl,
                    lslc * (sp2 - sl2) + (lc2 - ls2) * spl,
                    ls2 * sp2 + lc2 * sl2 + 2 * lslc * spl);
}

float TrapezoidalStripTopology::strip(const LocalPoint& lp) const {
  float aStrip = ((lp.x() * theDistToBeam / (theYAxOr * lp.y() + theDistToBeam)) - theOffset) / thePitch;
  if (aStrip < 0)
    aStrip = 0;
  else if (aStrip > theNumberOfStrips)
    aStrip = theNumberOfStrips;
  return aStrip;
}

MeasurementPoint TrapezoidalStripTopology::measurementPosition(const LocalPoint& lp) const {
  return MeasurementPoint(((lp.x() * theDistToBeam / (theYAxOr * lp.y() + theDistToBeam)) - theOffset) / thePitch,
                          lp.y() / theDetHeight);
}

MeasurementError TrapezoidalStripTopology::measurementError(const LocalPoint& lp, const LocalError& lerr) const {
  float lt, lc2, ls2, lslc;
  float localL, localP;
  float sl2, sp2, spl;
  lt = -lp.x() / (theYAxOr * lp.y() + theDistToBeam) * theYAxOr;
  lc2 = 1. / (1. + lt * lt);
  lslc = lt * lc2;
  ls2 = 1. - lc2;
  localL = theDetHeight / std::sqrt(lc2);
  localP = localPitch(lp);
  sp2 = lc2 * lerr.xx() + ls2 * lerr.yy() + 2 * lslc * lerr.xy();
  sl2 = ls2 * lerr.xx() + lc2 * lerr.yy() - 2 * lslc * lerr.xy();
  spl = lslc * (lerr.yy() - lerr.xx()) + (lc2 - ls2) * lerr.xy();
  return MeasurementError(sp2 / (localP * localP), spl / (localP * localL), sl2 / (localL * localL));
}

int TrapezoidalStripTopology::channel(const LocalPoint& lp) const {
  return std::min(int(strip(lp)), theNumberOfStrips - 1);
}

float TrapezoidalStripTopology::pitch() const { return thePitch; }

float TrapezoidalStripTopology::localPitch(const LocalPoint& lp) const {
  float x = lp.x();
  float y = theYAxOr * lp.y() + theDistToBeam;
  return thePitch * y / (theDistToBeam * std::sqrt(1.f + x * x / (y * y)));
}

float TrapezoidalStripTopology::stripAngle(float strip) const {
  return std::atan(-(strip * thePitch + theOffset) * theYAxOr / theDistToBeam);
}

int TrapezoidalStripTopology::nstrips() const { return theNumberOfStrips; }

float TrapezoidalStripTopology::shiftOffset(float pitch_fraction) {
  theOffset += thePitch * pitch_fraction;
  return theOffset;
}

float TrapezoidalStripTopology::localStripLength(const LocalPoint& lp) const {
  float ltan = -lp.x() / (theYAxOr * lp.y() + theDistToBeam) * theYAxOr;
  float localL = theDetHeight * std::sqrt(1.f + ltan * ltan);
  //  float lcos2 = 1.f/(1.f+ltan*ltan);
  //  float localL = theDetHeight / std::sqrt(lcos2);

  return localL;
}
