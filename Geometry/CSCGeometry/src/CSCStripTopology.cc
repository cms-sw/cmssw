#include "Geometry/CSCGeometry/interface/CSCStripTopology.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>

CSCStripTopology::CSCStripTopology(int ns, float aw, float dh, float r, float aoff, float ymid)
    : OffsetRadialStripTopology(ns, aw, dh, r, aoff, ymid) {}

CSCStripTopology::~CSCStripTopology() {}

std::pair<float, float> CSCStripTopology::equationOfStrip(float strip) const {
  const float fprec = 1.E-06;

  // slope of strip
  float strangle = M_PI_2 - stripAngle(strip);
  float ms = 0;
  if (fabs(strangle) > fprec)
    ms = tan(strangle);

  // intercept of strip
  float cs = -originToIntersection();

  LogTrace("CSCStripTopology|CSC") << "CSCStripTopology: strip=" << strip << ", strip angle = " << strangle
                                   << ", intercept on y axis=" << cs;

  return std::pair<float, float>(ms, cs);
}

std::pair<float, float> CSCStripTopology::yLimitsOfStripPlane() const {
  // use functions from base class
  float ylow = yCentreOfStripPlane() - yExtentOfStripPlane() / 2.;
  float yhigh = yCentreOfStripPlane() + yExtentOfStripPlane() / 2.;

  return std::pair<float, float>(ylow, yhigh);
}

// op<< is not a member

#include <iostream>

std::ostream& operator<<(std::ostream& os, const CSCStripTopology& st) {
  st.put(os) << " isa " << static_cast<const OffsetRadialStripTopology&>(st);
  return os;
}
