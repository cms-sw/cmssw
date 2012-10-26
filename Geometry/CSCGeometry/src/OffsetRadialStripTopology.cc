// This is OffsetRadialStripTopology.cc

// Implementation and use of base class RST depends crucially on stripAngle() being defined
// to include the angular strip offset relative to the local coordinate frame, while
// phiOfOneEdge() remains unchanged from its RST interpretation (i.e. phiOfOneEdge remains
// measured relative to the symmetry axis of the strip plane even in the OffsetRST case.)
// To help understand the implementation below I use the notation 'prime' for the local
// coordinate system rotated from the true local coordinate system by the angular offset
// of the offset RST. Thus the 'prime' system is aligned with the symmetry axes of the
// strip plane of the ORST.
// The following functions in the base class RST work fine for the ORST too since
// they're implemented in terms of stripAngle() and this is overridden for the ORST
// so that the angular offset is included:
//  xOfStrip(.)
//  localPitch(.)
//  localError(.,.)
//  localError(.,.)
//  measurementError(.,.)

#include <Geometry/CSCGeometry/interface/OffsetRadialStripTopology.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>
#include <cmath>

OffsetRadialStripTopology::OffsetRadialStripTopology( 
  int numberOfStrips, float stripPhiPitch,
  float detectorHeight, float radialDistance,
  float stripOffset, float yCentre ) :
  CSCRadialStripTopology( numberOfStrips, stripPhiPitch, detectorHeight, radialDistance, +1, yCentre),
	     theStripOffset( stripOffset )
{ 
  float rotate_by = stripOffset * angularWidth(); // now in angular units (radians, I hope)
  theCosOff = cos(rotate_by);
  theSinOff = sin(rotate_by);

  LogTrace("CSCStripTopology|CSC") << "fractional strip offset = " << stripOffset <<
    "\n angle = " << rotate_by << 
    " cos = " << theCosOff << " sin = " << theSinOff;
}

LocalPoint OffsetRadialStripTopology::localPosition(const MeasurementPoint & mp) const {
  // Local coordinates are (x,y). Coordinates along symmetry axes of strip
  // plane are (x',y'). These are rotated w.r.t. (x,y)

  // You might first think you could implement this as follows (cf. measurementPosition below):
  //  LocalPoint lpp = RST::localPosition(MP);
  //  return this->toLocal(lpp);
  // But this does not work because RST::localPosition makes use of stripAngle() - virtual - and thus
  // this is not the appropriate angle - it has the offset added, which is unwanted in this case!
  // So have to implement it directly...

  // 1st component of MP measures angular position within strip plane
  float phi = phiOfOneEdge() + mp.x() * angularWidth();
  // 2nd component of MP is fractional position along strip, with range +/-0.5,
  // so distance along strip, measured from mid-point of length of strip, is
  //     mp.y() * (length of strip). 
  // Distance in direction of coordinate y' is
  //    mp.y() * (length of strip) * cos(phi)
  // where phi is angle between strip and y' axis.
  // But (length of strip) = detHeight/cos(phi), so
  float yprime =  mp.y() * detHeight() + yCentreOfStripPlane();
  float xprime = ( originToIntersection() + yprime ) * tan ( phi );
  //  Rotate to (x,y)
  return toLocal(xprime, yprime);
}

MeasurementPoint OffsetRadialStripTopology::measurementPosition( const LocalPoint& lp ) const {
  LocalPoint lpp = this->toPrime(lp); // in prime system, aligned with strip plane sym axes
  return CSCRadialStripTopology::measurementPosition(lpp); // now can use the base class method
}

float OffsetRadialStripTopology::strip(const LocalPoint& lp) const {
  LocalPoint pnt = toPrime(lp);
  float phi = atan2( pnt.x(), pnt.y()+originToIntersection() );
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
             + originToIntersection() * theSinOff;
  float y = -theSinOff * xprime + theCosOff * yprime
             - originToIntersection() * (1. - theCosOff);
  return LocalPoint(x, y);
}

LocalPoint OffsetRadialStripTopology::toPrime(const LocalPoint& lp) const {
  float xprime = theCosOff * lp.x() - theSinOff * lp.y()
                  - originToIntersection() * theSinOff;
  float yprime = theSinOff * lp.x() + theCosOff * lp.y()
                  - originToIntersection() * (1. - theCosOff);
  return LocalPoint(xprime, yprime);
}

std::ostream & operator<<(std::ostream & os, const OffsetRadialStripTopology & orst)
{
  os << "OffsetRadialStripTopology isa "
     <<  static_cast<const CSCRadialStripTopology&>( orst )
     << "fractional strip offset   " << orst.stripOffset()
     << "\ncos(angular offset)       " << orst.theCosOff
     << "\nsin(angular offset)       " << orst.theSinOff << std::endl;
  return os;
}

