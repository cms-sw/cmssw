#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"

#include <iostream>
#include <cmath>
#include <algorithm>

RectangularStripTopology::RectangularStripTopology(int ns, float p, float l) : 
  thePitch(p), theNumberOfStrips(ns), theStripLength(l) { 
  theOffset = -0.5f*theNumberOfStrips * thePitch;

#ifdef VERBOSE
  cout <<"Constructing RectangularStripTopology with"
       <<" nstrips = "<<ns
       <<" pitch = "<<p
       <<" length = "<<l
       <<endl;
#endif
}
  
LocalPoint 
RectangularStripTopology::localPosition(float strip) const {
  return LocalPoint( strip*thePitch + theOffset, 0.0f);
}

LocalPoint 
RectangularStripTopology::localPosition(const MeasurementPoint& mp) const {
  return LocalPoint( mp.x()*thePitch+theOffset, mp.y()*theStripLength);
}

LocalError 
RectangularStripTopology::localError(float /*strip*/, float stripErr2) const{
  return LocalError(stripErr2 * thePitch*thePitch,
		    0.f,
		    theStripLength*theStripLength*(1.f/12.f));
}

LocalError 
RectangularStripTopology::localError(const MeasurementPoint& /*mp*/,
  const MeasurementError& merr) const{
  return LocalError(merr.uu() * thePitch*thePitch,
		    merr.uv() * thePitch*theStripLength,
		    merr.vv() * theStripLength*theStripLength);
}
  
float 
RectangularStripTopology::strip(const LocalPoint& lp) const {
  float aStrip = (lp.x() - theOffset) / thePitch;
  if (aStrip < 0 ) aStrip = 0;
  else if (aStrip > theNumberOfStrips)  aStrip = theNumberOfStrips;
  return aStrip;
}

MeasurementPoint 
RectangularStripTopology::measurementPosition(const LocalPoint& lp) const {
  return MeasurementPoint((lp.x()-theOffset)/thePitch,
                          lp.y()/theStripLength);
}

MeasurementError 
RectangularStripTopology::measurementError(const LocalPoint& /*lp*/,
  const LocalError& lerr) const {
  return MeasurementError(lerr.xx()/(thePitch*thePitch),
  			  lerr.xy()/(thePitch*theStripLength),
			  lerr.yy()/(theStripLength*theStripLength));
}

int 
RectangularStripTopology::channel(const LocalPoint& lp) const {
  return std::min(int(strip(lp)),theNumberOfStrips-1);
}

float 
RectangularStripTopology::pitch() const { 
  return thePitch;
}
  
float 
RectangularStripTopology::localPitch(const LocalPoint& /*lp*/) const {
  return thePitch;
}
  
float 
RectangularStripTopology::stripAngle(float /*strip*/) const {
  return 0;
}
  
int 
RectangularStripTopology::nstrips() const { 
  return theNumberOfStrips;
}

