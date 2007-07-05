#include <Geometry/CommonTopologies/interface/RadialStripTopology.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <Utilities/General/interface/CMSexception.h>

#include <iostream>
#include <cmath>
#include <algorithm>

RadialStripTopology::RadialStripTopology(int ns, float aw, float dh, float r, int yAx, float yMid) :
  theNumberOfStrips(ns), theAngularWidth(aw), 
  theDetHeight(dh), theCentreToIntersection(r),
  theYAxisOrientation(yAx), yCentre( yMid) {   
  // Angular offset of extreme edge of detector, so that angle is
  // zero for a strip lying along local y axis = long symmetry axis of plane of strips
  thePhiOfOneEdge = -(theNumberOfStrips/2.) * theAngularWidth * yAx;
  
  LogTrace("RadialStripTopology") << "RadialStripTopology: constructed with"
        << " strips = " << ns
        << " width = " << aw << " rad "
        << " det_height = " << dh
        << " ctoi = " << r 
        << " phi_edge = " << thePhiOfOneEdge << " rad "
        << " y_ax_ori = " << theYAxisOrientation
	<< " y_det_centre = " << yCentre 
        << "\n";
}    

float 
RadialStripTopology::xOfStrip(int strip, float y) const {
  // Expect input 'strip' to be in range 1 to nstrips()
  float tanPhi = tan( stripAngle(static_cast<float>(strip) - 0.5 ) );
  return yAxisOrientation()* yDistanceToIntersection( y ) * tanPhi;
}

LocalPoint 
RadialStripTopology::localPosition(float strip) const {
  return LocalPoint( yAxisOrientation() * originToIntersection() * tan( stripAngle(strip) ), 0.0 );
}

LocalPoint 
RadialStripTopology::localPosition(const MeasurementPoint& mp) const {
  float phi = stripAngle( mp.x() );
  // 2nd component of MP is fractional position along strip, range -0.5 to +0.5 in direction of y
  // so distance along strip (measured from mid-point of length of strip) is
  //     mp.y() * (length of strip). 
  // Thus distance along y direction, from mid-point of strip, is 
  //    mp.y() * (length of strip) * cos(phi)
  // But (length of strip) = theDetHeight/cos(phi), so
  float y =  mp.y() * detHeight() + yCentreOfStripPlane();
  float x = yAxisOrientation() * yDistanceToIntersection( y ) * tan ( phi );
  return LocalPoint( x, y );
}

LocalError 
RadialStripTopology::localError(float strip, float stripErr2) const {
  // Consider measurement as strip (phi) position and mid-point of strip
  // Since 'strip' is in units of angular strip-widths, stripErr2 is
  // required to be in corresponding units.

  float t = tan( stripAngle( strip ) );        // tan(angle between strip and y)
  float c2 = 1./(1. + t*t);                    // cos(angle)**2
  float cs = t*c2;                             // sin(angle)*cos(angle); tan carries sign of sin!
  float s2 = t*t * c2;                         // sin(angle)**2

  float D2 = centreToIntersection()*centreToIntersection() / c2; // (mid pt of strip to intersection)**2
  float L2 = detHeight()*detHeight() / c2;   // length**2 of strip across detector
  float A2 = angularWidth()*angularWidth();

  // The error**2 we're assigning is L2/12 wherever we measure the position along the strip
  // from... we just know the measurement is somewhere ON the strip. 

  float SD2 = L2 / 12.;       // SD = Sigma-Distance-along-strip
  float SA2 = A2 * stripErr2; // SA = Sigma-Angle-of-strip

  float sx2 = c2 * D2 * SA2 + s2 * SD2;
  float sy2 = s2 * D2 * SA2 + c2 * SD2;
  float rhosxsy = cs * ( SD2 - D2 * SA2 );

  return LocalError(sx2, rhosxsy, sy2);
}

LocalError 
RadialStripTopology::localError(const MeasurementPoint& mp, 
                                const MeasurementError& merr) const {
  // Here we need to allow the possibility of correlated errors, since
  // that may happen during Kalman filtering

  //  float phi = phiOfOneEdge() + yAxisOrientation() * mp.x() * angularWidth();
  float phi = stripAngle( mp.x() );

  //  float t = tan( phi );                        // tan(angle between strip and y)
  //float c2 = 1./(1. + t*t);                    // cos(angle)**2
  // float cs = t*c2;                             // sin(angle)*cos(angle); tan carries sign of sin!
  // float s2 = t*t * c2;                         // sin(angle)**2
  

  float s1 = sin(phi);
  float s2 = s1*s1;
  float c2 = 1.-s2;
  float c1 = std::sqrt(c2);
  float cs = c1*s1;

  float A  = angularWidth();
  float A2 = A * A;

  // D is distance from intersection of edges to hit on strip
  float D = (centreToIntersection() + yAxisOrientation() * mp.y() * detHeight()) /c1;
  float D2 = D * D;

  // L is length of strip across face of chamber
  float L = detHeight()/ c1;  
  float L2  = L*L; 

  // MeasurementError elements are already squared
  // but they're normalized to products of A and L 
  // (N.B. uses L=length of strip, and not D=distance to intersection)
  // Remember to ensure measurement error components are indeed normalized like this!

  float SA2 = merr.uu() * A2;
  float SD2 = merr.vv() * L2; // Note this norm uses stripLength**2
  float RHOSASR = merr.uv() * A * L;

  float sx2 = SA2 * D2 * c2  +  2. * RHOSASR * D * cs  +  SD2 * s2;
  float sy2 = SA2 * D2 * s2  -  2. * RHOSASR * D * cs  +  SD2 * c2;
  float rhosxsy = cs * ( SD2 - D2 * SA2 )  +  RHOSASR * D * ( c2 - s2 );

  return LocalError(sx2, rhosxsy, sy2);
}

float 
RadialStripTopology::strip(const LocalPoint& lp) const {
  // Note that this phi is measured from y axis, so sign of angle <-> sign of x * yAxisOrientation
  // Use atan2(x,y) rather than more usual atan2(y,x)

  float phi = atan2( lp.x(), yDistanceToIntersection( lp.y() ) );
  float aStrip = ( phi - yAxisOrientation() * phiOfOneEdge() )/angularWidth();
  if (aStrip < 0. ) aStrip = 0.;
  else if (aStrip > theNumberOfStrips)  aStrip = theNumberOfStrips;
  return aStrip;
}
 
MeasurementPoint 
RadialStripTopology::measurementPosition(const LocalPoint& lp) const {
  // Note that this phi is (pi/2 - conventional local phi)
  // This means use atan2(x,y) rather than more usual atan2(y,x)
  float phi = yAxisOrientation() * atan2( lp.x(), yDistanceToIntersection( lp.y() ) );
  return MeasurementPoint( yAxisOrientation()*( phi-phiOfOneEdge() )/angularWidth(),
                          (lp.y() - yCentreOfStripPlane())/detHeight() );
}

MeasurementError 
RadialStripTopology::measurementError(const LocalPoint& lp,
  const LocalError& lerr) const {

  float yHitToInter = yDistanceToIntersection( lp.y() );
  float t  = yAxisOrientation() * lp.x() / yHitToInter; // tan(angle between strip and y) 
  float c2 = 1./(1. + t*t);  // cos(angle)**2
  float cs = t*c2;           // sin(angle)*cos(angle); tan carries sign of sin!
  float s2 = 1. - c2;        // sin(angle)**2

  float A  = angularWidth();

  // L is length of strip across face of chamber
  float L2 = detHeight()*detHeight() / c2;  
  float L  = sqrt(L2); 

  // D is distance from intersection of edges to hit on strip
  float D2 = lp.x()*lp.x() + yHitToInter*yHitToInter;
  float D = sqrt(D2);

  float LP = D * A;
  float LP2 = LP * LP;

  float SA2 = ( c2 * lerr.xx() - 2. * cs * lerr.xy() + s2 * lerr.yy() ) / LP2;
  float SD2 = ( s2 * lerr.xx() + 2. * cs * lerr.xy() + c2 * lerr.yy() ) / L2;
  float RHOSASR = ( cs * ( lerr.xx() - lerr.yy() ) + ( c2 - s2 ) * lerr.xy() ) / (LP*L);

  return MeasurementError(SA2, RHOSASR, SD2);
}

int 
RadialStripTopology::channel(const LocalPoint& lp) const {
  return std::min( int( strip(lp) ), theNumberOfStrips-1 );
}

float 
RadialStripTopology::pitch() const { 
  // BEWARE: this originally returned the pitch at the local origin 
  // but Tracker prefers throwing an exception since the strip is 
  // not constant along local x axis for a RadialStripTopology.

  throw Genexception("RadialStripTopology::pitch() called - makes no sense, use localPitch(.) instead.");
  return 0.;
}
  
float 
RadialStripTopology::localPitch(const LocalPoint& lp) const {
  // The local pitch is the local x width of the strip at the local (x,y)

  // Calculating it is a nightmare...deriving the expression below is left
  // as a exercise for the reader. 

  float fstrip = strip(lp); // position in strip units
  int istrip = static_cast<int>(fstrip) + 1; // which strip number
  if (istrip>nstrips() ) istrip = nstrips(); // enforce maximum
  float fangle = stripAngle(static_cast<float>(istrip) - 0.5); // angle of strip centre
  float localp = yDistanceToIntersection( lp.y() ) * sin(angularWidth()) /
    ( cos(fangle-0.5*angularWidth())*cos(fangle+0.5*angularWidth()) );
  return localp;
}
  
float 
RadialStripTopology::stripAngle(float strip) const {
  return ( phiOfOneEdge() + yAxisOrientation() * strip * angularWidth() );
}
  
float RadialStripTopology::localStripLength(const LocalPoint& lp) const {
  float yHitToInter = yDistanceToIntersection( lp.y() );
  // since we're dealing with magnitudes, sign is unimportant
  float t  = lp.x() / yHitToInter;    // tan(angle between strip and y)
  float c2 = 1./(1. + t*t);           // cos(angle)**2
  return detHeight() / sqrt(c2);
}

int RadialStripTopology::nearestStrip(const LocalPoint & lp) const
{
  // xxxStripTopology::strip() is expected to have range 0. to 
  // float(no of strips), but be extra careful and enforce that

  float fstrip = this->strip(lp);
  int n_strips = this->nstrips();
  fstrip = ( fstrip>=0. ? fstrip :  0. ); // enforce minimum 0.
  int near = static_cast<int>( fstrip ) + 1; // first strip is 1
  near = ( near<=n_strips ? near : n_strips ); // enforce maximum at right edge
  return near;
}

float RadialStripTopology::yDistanceToIntersection( float y ) const {
  return yAxisOrientation() * y + originToIntersection();
}

std::ostream & operator<<( std::ostream & os, const RadialStripTopology & rst )
{
  os  << "RadialStripTopology " << std::endl
      << " " << std::endl
      << "number of strips          " << rst.nstrips() << std::endl
      << "centre to whereStripsMeet " << rst.centreToIntersection() << std::endl
      << "detector height in y      " << rst.detHeight() << std::endl
      << "angular width of strips   " << rst.phiPitch() << std::endl
      << "phi of one edge           " << rst.phiOfOneEdge() << std::endl
      << "y axis orientation        " << rst.yAxisOrientation() << std::endl
      << "y of centre of strip plane " << rst.yCentreOfStripPlane() << std::endl;
  return os;
}
