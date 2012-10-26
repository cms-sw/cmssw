#include <Geometry/CommonTopologies/interface/CSCRadialStripTopology.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <Utilities/General/interface/CMSexception.h>

#include <iostream>
#include <cmath>
#include <algorithm>

CSCRadialStripTopology::CSCRadialStripTopology(int ns, float aw, float dh, float r, float yAx, float yMid) :
  theNumberOfStrips(ns), theAngularWidth(aw), 
  theDetHeight(dh), theCentreToIntersection(r),
  theYAxisOrientation(yAx), yCentre( yMid) {   
  // Angular offset of extreme edge of detector, so that angle is
  // zero for a strip lying along local y axis = long symmetry axis of plane of strips
  thePhiOfOneEdge = -(0.5*theNumberOfStrips) * theAngularWidth * yAx;
  
  LogTrace("CSCRadialStripTopology") << "CSCRadialStripTopology: constructed with"
        << " strips = " << ns
        << " width = " << aw << " rad "
        << " det_height = " << dh
        << " ctoi = " << r 
        << " phi_edge = " << thePhiOfOneEdge << " rad "
        << " y_ax_ori = " << theYAxisOrientation
	<< " y_det_centre = " << yCentre 
        << "\n";
}    

int CSCRadialStripTopology::channel(const LocalPoint& lp) const { return   std::min( int( strip(lp) ), theNumberOfStrips-1 ) ;}

int CSCRadialStripTopology::nearestStrip(const LocalPoint & lp) const {   return std::min( nstrips(), static_cast<int>( std::max(float(0), strip(lp)) ) + 1);}

float CSCRadialStripTopology::stripAngle(float strip) const { return   phiOfOneEdge() + yAxisOrientation() * strip * angularWidth() ;}

float CSCRadialStripTopology::yDistanceToIntersection( float y ) const { return   yAxisOrientation()*y + originToIntersection() ;}

float CSCRadialStripTopology::localStripLength(const LocalPoint& lp) const {  
  return detHeight() * std::sqrt(1.f + std::pow( lp.x()/yDistanceToIntersection(lp.y()), 2.f) );
}

float CSCRadialStripTopology::xOfStrip(int strip, float y) const { 
  return   
    yAxisOrientation() * yDistanceToIntersection( y ) * std::tan( stripAngle(static_cast<float>(strip) - 0.5 ) );
}

float CSCRadialStripTopology::strip(const LocalPoint& lp) const {
  const float   // phi is measured from y axis --> sign of angle is sign of x * yAxisOrientation --> use atan2(x,y), not atan2(y,x)
    phi( std::atan2( lp.x(), yDistanceToIntersection( lp.y() ) )),
    aStrip( ( phi - yAxisOrientation() * phiOfOneEdge() )/angularWidth());
  return  std::max(float(0), std::min( (float)nstrips(), aStrip ));
}

LocalPoint CSCRadialStripTopology::localPosition(float strip) const {
  return LocalPoint( yAxisOrientation() * originToIntersection() * tan( stripAngle(strip) ), 0 );
}

LocalPoint CSCRadialStripTopology::localPosition(const MeasurementPoint& mp) const {
  const float  // y = (L/cos(phi))*mp.y()*cos(phi) 
    y( mp.y()*detHeight()  +  yCentreOfStripPlane() ),
    x( yAxisOrientation() * yDistanceToIntersection( y ) * std::tan ( stripAngle( mp.x() ) ) );
  return LocalPoint( x, y );
}

MeasurementPoint CSCRadialStripTopology::measurementPosition(const LocalPoint& lp) const {
  const float // phi is [pi/2 - conventional local phi], use atan2(x,y) rather than atan2(y,x)
    phi( yAxisOrientation() * std::atan2( lp.x(), yDistanceToIntersection( lp.y() ) ));
  return MeasurementPoint( yAxisOrientation()*( phi-phiOfOneEdge() ) / angularWidth(),
			   ( lp.y() - yCentreOfStripPlane() )        / detHeight() );
}

LocalError CSCRadialStripTopology::localError(float strip, float stripErr2) const {
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

LocalError CSCRadialStripTopology::localError(const MeasurementPoint& mp, const MeasurementError& me) const {
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

MeasurementError CSCRadialStripTopology::measurementError(const LocalPoint& p,  const LocalError& e) const {
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
 

float CSCRadialStripTopology::localPitch(const LocalPoint& lp) const { 
 // The local pitch is the local x width of the strip at the local (x,y)
  const int istrip = std::min(nstrips(), static_cast<int>(strip(lp)) + 1); // which strip number
  const float fangle = stripAngle(static_cast<float>(istrip) - 0.5); // angle of strip centre
  return
    yDistanceToIntersection( lp.y() ) * std::sin(angularWidth()) /
    std::pow( std::cos(fangle-0.5f*angularWidth()), 2.f);
}
