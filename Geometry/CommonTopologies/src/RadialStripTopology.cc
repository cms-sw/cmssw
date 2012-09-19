#include <Geometry/CommonTopologies/interface/RadialStripTopology.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <Utilities/General/interface/CMSexception.h>

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>


namespace {

  // valid for z < pi/8
  inline 
  float atan0(float z) {
    assert(z<0.39);
    float z2 = z * z;
    return
      ((( 8.05374449538e-2f * z2
	  - 1.38776856032E-1f) * z2
	+ 1.99777106478E-1f) * z2
       - 3.33329491539E-1f) * z2 * z
      + z;
  }

}


RadialStripTopology::RadialStripTopology(int ns, float aw, float dh, float r, int yAx, float yMid) :
  theNumberOfStrips(ns), theAngularWidth(aw), 
  theDetHeight(dh), theCentreToIntersection(r),
  theYAxisOrientation(yAx), yCentre( yMid) {   
  // Angular offset of extreme edge of detector, so that angle is
  // zero for a strip lying along local y axis = long symmetry axis of plane of strips
  thePhiOfOneEdge = -(0.5*theNumberOfStrips) * theAngularWidth; // always negative!
  
  assert(std::abs(thePhiOfOneEdge)<0.35); // < pi/8 (and some tollerance)

  std::cout << "VI: work in progress :RadialStripTopology buggy" << std::endl;

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

int RadialStripTopology::channel(const LocalPoint& lp) const { return   std::min( int( strip(lp) ), theNumberOfStrips-1 ) ;}

int RadialStripTopology::nearestStrip(const LocalPoint & lp) const {   return std::min( nstrips(), static_cast<int>( std::max(float(0), strip(lp)) ) + 1);}

float RadialStripTopology::stripAngle(float strip) const { return   yAxisOrientation() * (phiOfOneEdge() +  strip * angularWidth()) ;}

float RadialStripTopology::yDistanceToIntersection( float y ) const { return   yAxisOrientation()*y + originToIntersection() ;}

float RadialStripTopology::localStripLength(const LocalPoint& lp) const {  
  return detHeight() * std::sqrt(1.f + std::pow( lp.x()/yDistanceToIntersection(lp.y()), 2.f) );
}

float RadialStripTopology::xOfStrip(int strip, float y) const { 
  return   
    yAxisOrientation() * yDistanceToIntersection( y ) * std::tan( stripAngle(static_cast<float>(strip) - 0.5 ) );
}

float RadialStripTopology::strip(const LocalPoint& lp) const {
     // phi is measured from y axis --> sign of angle is sign of x * yAxisOrientation --> use atan2(x,y), not atan2(y,x)
  float y = yDistanceToIntersection( lp.y() );
  const float phi =  std::copysign(atan0(std::abs(lp.x()/y)) , lp.x() );
  const float aStrip = ( phi -  phiOfOneEdge() )/angularWidth();
  return  std::max(float(0), std::min( (float)nstrips(), aStrip ));
}

LocalPoint RadialStripTopology::localPosition(float strip) const {
  return LocalPoint( yAxisOrientation() * originToIntersection() * tan( stripAngle(strip) ), 0 );
}

LocalPoint RadialStripTopology::localPosition(const MeasurementPoint& mp) const {
  const float  // y = (L/cos(phi))*mp.y()*cos(phi) 
    y( mp.y()*detHeight()  +  yCentreOfStripPlane() ),
    x( yAxisOrientation() * yDistanceToIntersection( y ) * std::tan ( stripAngle( mp.x() ) ) );
  return LocalPoint( x, y );
}

MeasurementPoint RadialStripTopology::measurementPosition(const LocalPoint& lp) const {
  // phi is [pi/2 - conventional local phi], use atan2(x,y) rather than atan2(y,x)
  const float phi = std::copysign(atan0(std::abs( lp.x()/yDistanceToIntersection(lp.y()) )) , lp.x() );
  return MeasurementPoint( ( phi-phiOfOneEdge() ) / angularWidth(),
			   ( lp.y() - yCentreOfStripPlane() )        / detHeight() );
}

LocalError RadialStripTopology::localError(float strip, float stripErr2) const {
  const double
    phi(stripAngle(strip)), t1(std::tan(phi)), t2(t1*t1),
    // s1(std::sin(phi)), c1(std::cos(phi)),
    // cs(s1*c1), s2(s1*s1), c2(1-s2), // rotation matrix

    tt( stripErr2 * std::pow( centreToIntersection()*angularWidth() ,2.f) ), // tangential sigma^2   *c2
    rr( std::pow(detHeight(), 2.f) * (1.f/12.f) ),                                   // radial sigma^2( uniform prob density along strip)  *c2

    xx( tt + t2*rr  ),
    yy( t2*tt + rr  ),
    xy( t1*( rr - tt ) );
  
  return LocalError( xx, xy, yy );
}

LocalError RadialStripTopology::localError(const MeasurementPoint& mp, const MeasurementError& me) const {
  const double
    phi(stripAngle(mp.x())), s1(std::sin(phi)), c1(std::cos(phi)),
    cs(s1*c1), s2(s1*s1), c2(1-s2), // rotation matrix

    T( angularWidth() * ( centreToIntersection() + yAxisOrientation()*mp.y()*detHeight()) / c1 ), // tangential measurement unit (local pitch)
    R( detHeight()/ c1 ),  // radial measurement unit (strip length)
    tt( me.uu() * T*T ),   // tangential sigma^2
    rr( me.vv() * R*R   ), // radial sigma^2
    tr( me.uv() * T*R ),  
    
    xx(  c2*tt  +  2*cs*tr  +  s2*rr      ),
    yy(  s2*tt  -  2*cs*tr  +  c2*rr      ),
    xy( cs*( rr - tt )  +  tr*( c2 - s2 ) );

  return LocalError( xx, xy, yy );
}

MeasurementError RadialStripTopology::measurementError(const LocalPoint& p,  const LocalError& e) const {
  const double
    yHitToInter(yDistanceToIntersection(p.y())),
    t(yAxisOrientation() * p.x() / yHitToInter),   // tan(strip angle) 
    cs(t/(1+t*t)), s2(t*cs), c2(1-s2),             // rotation matrix

    T2( 1./(std::pow(angularWidth(),2.f) * ( std::pow(p.x(),2.f) + std::pow(yHitToInter,2)) )), // 1./tangential measurement unit (local pitch) ^2
    R2( c2/std::pow(detHeight(),2.f) ),                                    // 1./ radial measurement unit (strip length) ^2

    uu(       ( c2*e.xx() - 2*cs*e.xy() + s2*e.yy() )   * T2 ),
    vv(       ( s2*e.xx() + 2*cs*e.xy() + c2*e.yy() )   * R2 ),
    uv( ( cs*( e.xx() - e.yy() ) + e.xy()*( c2 - s2 ) )  * std::sqrt (T2*R2) );
  
  return MeasurementError(uu, uv, vv);
}
 
float RadialStripTopology::pitch() const {  throw Genexception("RadialStripTopology::pitch() called - makes no sense, use localPitch(.) instead."); return 0.;}

float RadialStripTopology::localPitch(const LocalPoint& lp) const { 
 // The local pitch is the local x width of the strip at the local (x,y)
  const int istrip = std::min(nstrips(), static_cast<int>(strip(lp)) + 1); // which strip number
  const float fangle = stripAngle(static_cast<float>(istrip) - 0.5); // angle of strip centre
  return
    yDistanceToIntersection( lp.y() ) * std::sin(angularWidth()) /
    std::pow( std::cos(fangle-0.5f*angularWidth()), 2.f);
}
    

std::ostream & operator<<( std::ostream & os, const RadialStripTopology & rst ) {
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
