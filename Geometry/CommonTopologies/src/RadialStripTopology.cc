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
  double tanPhi = tan( stripAngle(static_cast<float>(strip) - 0.5 ) );
  return yAxisOrientation()*( y*yAxisOrientation()+originToIntersection() ) * tanPhi;
}

LocalPoint 
RadialStripTopology::localPosition(float strip) const {
  return LocalPoint( yAxisOrientation() * originToIntersection() * tan( stripAngle(strip) ), 0.0 );
}

LocalPoint 
RadialStripTopology::localPosition(const MeasurementPoint& mp) const {
  double phi = stripAngle( mp.x() );
  // 2nd component of MP is fractional position along strip, range +/-0.5,
  // so distance along strip (measured from mid-point of length of strip) is
  //     mp.y() * (length of strip). 
  // Thus distance along y direction, from mid-point of strip, is 
  //    mp.y() * (length of strip) * cos(phi)
  // But (length of strip) = theDetHeight/cos(phi), so
  double y =  mp.y() * theDetHeight + yCentreOfStripPlane();
  double x = yAxisOrientation() * ( originToIntersection() + y * yAxisOrientation() ) * tan ( phi );
  return LocalPoint( x, y );
}

LocalError 
RadialStripTopology::localError(float strip, float stripErr2) const {
  // Consider measurement as strip (phi) position and mid-point of strip
  // Since 'strip' is in units of angular strip-widths, stripErr2 is
  // required to be in corresponding units.

  double t = tan( stripAngle( strip ) );        // tan(angle between strip and y)
  double c2 = 1./(1. + t*t);                    // cos(angle)**2
  double cs = t*c2;                             // sin(angle)*cos(angle); tan carries sign of sin!
  double s2 = t*t * c2;                         // sin(angle)**2

  double D2 = originToIntersection()*originToIntersection() / c2;
  double L2 = theDetHeight*theDetHeight / c2;   // length**2 of strip across detector
  double A2 = theAngularWidth*theAngularWidth;

  // The error**2 we're assigning is L2/12 wherever we measure the position along the strip
  // from... we just know the measurement is somewhere ON the strip. 

  double SD2 = L2 / 12.;       // SR = Sigma-Radius ('Radius' is along strip)
  double SA2 = A2 * stripErr2; // SA = Sigma-Angle

  double sx2 = c2 * D2 * SA2 + s2 * SD2;
  double sy2 = s2 * D2 * SA2 + c2 * SD2;
  double rhosxsy = cs * ( SD2 - D2 * SA2 );

  return LocalError(sx2, rhosxsy, sy2);
}

LocalError 
RadialStripTopology::localError(const MeasurementPoint& mp, 
                                const MeasurementError& merr) const {
  // Here we need to allow the possibility of correlated errors, since
  // that may happen during Kalman filtering

  double phi = thePhiOfOneEdge + mp.x() * theAngularWidth;

  double t = tan( phi );                        // tan(angle between strip and y)
  double c2 = 1./(1. + t*t);                    // cos(angle)**2
  double cs = t*c2;                             // sin(angle)*cos(angle); tan carries sign of sin!
  double s2 = t*t * c2;                         // sin(angle)**2

  double A  = theAngularWidth;
  double A2 = A * A;

  // D is distance from intersection of edges to hit on strip
  double D = (originToIntersection() + mp.y() * yAxisOrientation() * theDetHeight) / sqrt(c2);
  double D2 = D * D;

  // L is length of strip across face of chamber
  double L2 = theDetHeight*theDetHeight / c2;  
  double L  = sqrt(L2); 

  // MeasurementError elements are already squared
  // but they're normalized to products of A and L (N.B. L not D!)
  // @@ ENSURE MEASUREMENT ERROR COMPONENTS ARE INDEED NORMALIZED LIKE THIS!

  double SA2 = merr.uu() * A2;
  double SD2 = merr.vv() * L2; // Note this norm uses stripLength**2
  double RHOSASR = merr.uv() * A * L;

  double sx2 = SA2 * D2 * c2  +  2. * RHOSASR * D * cs  +  SD2 * s2;
  double sy2 = SA2 * D2 * s2  -  2. * RHOSASR * D * cs  +  SD2 * c2;
  double rhosxsy = cs * ( SD2 - D2 * SA2 )  +  RHOSASR * D * ( c2 - s2 );

  return LocalError(sx2, rhosxsy, sy2);
}

float 
RadialStripTopology::strip(const LocalPoint& lp) const {
  // Note that this phi is (pi/2 - conventional local phi)
  // This means use atan2(x,y) rather than more usual atan2(y,x)
  double phi = yAxisOrientation() * atan2( lp.x(), lp.y()*yAxisOrientation()+originToIntersection() );
  double aStrip = yAxisOrientation() * (phi-thePhiOfOneEdge)/theAngularWidth;
  aStrip = (aStrip >= 0. ? aStrip : 0.);
  aStrip = (aStrip <= theNumberOfStrips ? aStrip : theNumberOfStrips);
  return aStrip;
}
 
MeasurementPoint 
RadialStripTopology::measurementPosition(const LocalPoint& lp) const {
  // Note that this phi is (pi/2 - conventional local phi)
  // This means use atan2(x,y) rather than more usual atan2(y,x)
  double phi = yAxisOrientation() * atan2( lp.x(), lp.y()*yAxisOrientation()+originToIntersection() );
  return MeasurementPoint( yAxisOrientation()*(phi-thePhiOfOneEdge)/theAngularWidth,
			   lp.y()/theDetHeight );
}

MeasurementError 
RadialStripTopology::measurementError(const LocalPoint& lp,
				      const LocalError& lerr) const {

  double yHitToInter = lp.y()*yAxisOrientation() + originToIntersection();
  // Care! sign of angle measurement must be consistently treated when yAxis orientation changes.
  double t  = yAxisOrientation() * lp.x() / yHitToInter; // tan(angle between strip and y) 
  double c2 = 1./(1. + t*t);  // cos(angle)**2
  double cs = t*c2;           // sin(angle)*cos(angle); tan carries sign of sin!
  double s2 = t*t * c2;       // sin(angle)**2

  double A  = theAngularWidth;

  // L is length of strip across face of chamber
  double L2 = theDetHeight*theDetHeight / c2;  
  double L  = sqrt(L2); 

  // D is distance from intersection of edges to hit on strip
  double D2 = lp.x()*lp.x() + yHitToInter*yHitToInter;
  double D = sqrt(D2);

  double LP = D * A;
  double LP2 = LP * LP;

  double SA2 = ( c2 * lerr.xx() - 2. * cs * lerr.xy() + s2 * lerr.yy() ) / LP2;
  double SD2 = ( s2 * lerr.xx() + 2. * cs * lerr.xy() + c2 * lerr.yy() ) / L2;
  double RHOSASR = ( cs * ( lerr.xx() - lerr.yy() ) + ( c2 - s2 ) * lerr.xy() ) / LP / L;

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

  double fstrip = strip(lp); // position in strip units
  int istrip = static_cast<int>(fstrip + 1.0); // which strip number
  istrip = (istrip>nstrips() ? nstrips() : istrip); // enforce maximum
  double fangle = stripAngle(static_cast<float>(istrip - 0.5)); // angle of strip centre
  double localp = ( lp.y()*yAxisOrientation() + originToIntersection() ) * sin(theAngularWidth) /
    ( cos(fangle-theAngularWidth/2.)*cos(fangle+theAngularWidth/2.) );
  return localp;
}
  
float 
RadialStripTopology::stripAngle(float strip) const {
  return ( thePhiOfOneEdge + yAxisOrientation()*strip*theAngularWidth );
}
  
float RadialStripTopology::localStripLength(const LocalPoint& lp) const {
  double yHitToInter = lp.y()*yAxisOrientation() + originToIntersection();
  // since we're dealing with magnitudes, sign is unimportant
  double t  = lp.x() / yHitToInter;    // tan(angle between strip and y)
  double c2 = 1./(1. + t*t);           // cos(angle)**2
  return theDetHeight / sqrt(c2);
}

int RadialStripTopology::nearestStrip(const LocalPoint & lp) const
{
  // xxxStripTopology::strip() is expected to have range 0. to 
  // float(no of strips), but be extra careful and enforce that

  double fstrip = this->strip(lp);
  int n_strips = this->nstrips();
  fstrip = ( fstrip>=0. ? fstrip :  0. ); // enforce minimum 0.
  int near = static_cast<int>( fstrip ) + 1; // first strip is 1
  near = ( near<=n_strips ? near : n_strips ); // enforce maximum at right edge
  return near;
}

std::ostream & operator<<( std::ostream & os, const RadialStripTopology & rst )
{
  os  << "RadialStripTopology " << std::endl
      << " " << std::endl
      << "number of strips          " << rst.nstrips() << std::endl
      << "centre to whereStripsMeet " << rst.centreToIntersection() << std::endl
      << "detector height in y      " << rst.stripLength() << std::endl
      << "angular width of strips   " << rst.phiPitch() << std::endl
      << "phi of one edge           " << rst.thePhiOfOneEdge << std::endl
      << "y axis orientation        " << rst.yAxisOrientation() << std::endl
      << "y of centre of strip plane " << rst.yCentreOfStripPlane() << std::endl;
  return os;
}
