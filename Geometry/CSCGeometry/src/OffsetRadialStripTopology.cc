// This is OffsetRadialStripTopology.cc

#include <Geometry/CSCGeometry/interface/OffsetRadialStripTopology.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>
#include <cmath>

OffsetRadialStripTopology::OffsetRadialStripTopology( 
  int numberOfStrips, float stripPhiPitch,
  float detectorHeight, float radialDistance,
  float stripOffset):
  RadialStripTopology( numberOfStrips, stripPhiPitch,
		       detectorHeight, radialDistance),
	     theStripOffset( stripOffset )
{ 
  // Reset offset in base class
  //  float off = RadialStripTopology::shiftOffset( stripOffset );
  float rotate_by = stripOffset * angularWidth(); // now in angular units
  theCosOff = cos(rotate_by);
  theSinOff = sin(rotate_by);

  LogDebug("CSC") << "fractional strip offset = " << stripOffset <<
    "\n angle = " << rotate_by << 
    " cos = " << theCosOff << " sin = " << theSinOff << "\n";
}

LocalPoint OffsetRadialStripTopology::localPosition(const MeasurementPoint & mp) const {
  // Local coordinates are (x,y). Coordinates along symmetry axes of strip
  // plane are (x',y'). These are rotated w.r.t. (x,y)

  // 1st component of MP measures angular position within strip plane
  float phi = phiOfOneEdge() + mp.x() * angularWidth();
  // 2nd component of MP is fractional position along strip, with range +/-0.5,
  // so distance along strip, measured from mid-point of length of strip, is
  //     mp.y() * (length of strip). 
  // Strip is always bisected by y'=0, so perpendicular distance
  // from this axis, coordinate y' is
  //    mp.y() * (length of strip) * cos(phi)
  // where phi is angle between strip and y' axis.
  // But (length of strip) = detHeight/cos(phi), so
  float yprime =  mp.y() * detHeight();
  float xprime = ( centreToIntersection() + yprime ) * tan ( phi );
  //  Rotate to (x,y)
  return toLocal(xprime, yprime);
}

float OffsetRadialStripTopology::strip(const LocalPoint& lp) const {
  LocalPoint pnt = toPrime(lp);
  float phi = atan2( pnt.x(), pnt.y()+centreToIntersection() );
  float fstrip = ( phi - phiOfOneEdge() ) / angularWidth();
  fstrip = ( fstrip>=0. ? fstrip : 0. );
  fstrip = ( fstrip<=nstrips() ? fstrip : nstrips() );
  return fstrip;
}

float OffsetRadialStripTopology::stripAngle(float strip) const {
  return ( phiOfOneEdge() + (strip+theStripOffset)*angularWidth() );
}

LocalPoint OffsetRadialStripTopology::toLocal(float xprime, float yprime) const {
  float x =  theCosOff * xprime + theSinOff * yprime
             + centreToIntersection() * theSinOff;
  float y = -theSinOff * xprime + theCosOff * yprime
             - centreToIntersection() * (1. - theCosOff);
  return LocalPoint(x, y);
}

LocalPoint OffsetRadialStripTopology::toPrime(const LocalPoint& lp) const {
  float xprime = theCosOff * lp.x() - theSinOff * lp.y()
                  - centreToIntersection() * theSinOff;
  float yprime = theSinOff * lp.x() + theCosOff * lp.y()
                  - centreToIntersection() * (1. - theCosOff);
  return LocalPoint(xprime, yprime);
}

std::ostream & operator<<(std::ostream & os, const OffsetRadialStripTopology & orst)
{
  os << "OffsetRadialStripTopology isa "
     <<  static_cast<const RadialStripTopology&>( orst )
     << "fractional strip offset   " << orst.stripOffset()
     << "\ncos(angular offset)       " << orst.theCosOff
     << "\nsin(angular offset)       " << orst.theSinOff << std::endl;
  return os;
}

