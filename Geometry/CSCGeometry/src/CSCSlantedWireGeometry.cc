#include <Geometry/CSCGeometry/src/CSCSlantedWireGeometry.h>
//#include <Geometry/CSCSimAlgo/interface/CSCGeometryLogger.h>
#include <Geometry/Vector/interface/LocalPoint.h>
#include <Geometry/CSCGeometry/interface/nint.h>
#include <cmath>

CSCSlantedWireGeometry::CSCSlantedWireGeometry( double wireSpacing,
	     double yOfFirstWire, float wireAngle ) :
  CSCWireGeometry( wireSpacing, yOfFirstWire ), theWireAngle( wireAngle ){
    cosWireAngle = cos( wireAngle );
    sinWireAngle = sin( wireAngle );
    theWireOffset = yOfFirstWire * cosWireAngle; 
    //    std::cout <<
    //       "CSCSlantedWireGeometry: constructed:\n" <<
    //       " wireSpacing = " << wireSpacing << 
    //       ", yOfFirstWire = " << yOfFirstWire << 
    //       ", wireAngle = " << wireAngle << 
    //       ", theWireOffset = " << theWireOffset << std::endl;
}

int CSCSlantedWireGeometry::nearestWire(const LocalPoint& lp) const {
  // Return nearest wire number to input LocalPoint.
  // Beware this may not exist or be read out!

  // rotate point to an axis perp. to wires
  float yprime = lp.y() * cosWireAngle - lp.x() * sinWireAngle;

  // climb the ladder
  return 1 + nint( (yprime - theWireOffset) / wireSpacing() );
} 


float CSCSlantedWireGeometry::yOfWire(float wire, float x) const {
  // Return local y of given wire, at given x

  // y in rotated frame with y axis perpendicular to wires...
  float yprime = theWireOffset + (wire-1.) * wireSpacing();
  // then y in usual (unrotated!) local xy frame...
  return ( yprime + x*sinWireAngle )/cosWireAngle;
}
