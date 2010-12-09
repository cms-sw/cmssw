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

int RadialStripTopology::channel(const LocalPoint& lp) const { return   std::min( int( strip(lp) ), theNumberOfStrips-1 ) ;}
int RadialStripTopology::nearestStrip(const LocalPoint & lp) const {   return std::min( nstrips(), static_cast<int>( std::max(float(0), strip(lp)) ) + 1);}
float RadialStripTopology::stripAngle(float strip) const { return   phiOfOneEdge() + yAxisOrientation() * strip * angularWidth() ;}
float RadialStripTopology::yDistanceToIntersection( float y ) const { return   yAxisOrientation()*y + originToIntersection() ;}
float RadialStripTopology::localStripLength(const LocalPoint& lp) const {  return detHeight() * sqrt(1. + pow( lp.x()/yDistanceToIntersection(lp.y()), 2) );}
float RadialStripTopology::xOfStrip(int strip, float y) const { return   
    yAxisOrientation() * yDistanceToIntersection( y ) * tan( stripAngle(static_cast<float>(strip) - 0.5 ) );}

float RadialStripTopology::
strip(const LocalPoint& lp) const {
  const float   // phi is measured from y axis --> sign of angle is sign of x * yAxisOrientation --> use atan2(x,y), not atan2(y,x)
    phi( atan2( lp.x(), yDistanceToIntersection( lp.y() ) )),
    aStrip( ( phi - yAxisOrientation() * phiOfOneEdge() )/angularWidth());
  return  std::max(float(0), std::min( (float)nstrips(), aStrip ));
}

LocalPoint RadialStripTopology::
localPosition(float strip) const {
  return LocalPoint( yAxisOrientation() * originToIntersection() * tan( stripAngle(strip) ), 0.0 );
}

LocalPoint RadialStripTopology::
localPosition(const MeasurementPoint& mp) const {
  const float  // y = (L/cos(phi))*mp.y()*cos(phi) 
    y( mp.y()*detHeight()  +  yCentreOfStripPlane() ),
    x( yAxisOrientation() * yDistanceToIntersection( y ) * tan ( stripAngle( mp.x() ) ) );
  return LocalPoint( x, y );
}

MeasurementPoint RadialStripTopology::
measurementPosition(const LocalPoint& lp) const {
  const float // phi is [pi/2 - conventional local phi], use atan2(x,y) rather than atan2(y,x)
    phi( yAxisOrientation() * atan2( lp.x(), yDistanceToIntersection( lp.y() ) ));
  return MeasurementPoint( yAxisOrientation()*( phi-phiOfOneEdge() ) / angularWidth(),
			   ( lp.y() - yCentreOfStripPlane() )        / detHeight() );
}

LocalError RadialStripTopology::
localError(float strip, float stripErr2) const {
  const double
    t(tan(stripAngle(strip))),
    cs(t/(1+t*t)), s2(t*cs), c2(1-s2),  // rotation matrix

    tt( stripErr2 * pow( centreToIntersection()*angularWidth() ,2) / c2 ), // tangential sigma^2
    rr( pow(detHeight(), 2) / (12*c2) ),                                   // radial sigma^2( uniform prob density along strip)

    xx( c2*tt + s2*rr  ),
    yy( s2*tt + c2*rr  ),
    xy( cs*( rr - tt ) );
  
  return LocalError( xx, xy, yy );
}

LocalError RadialStripTopology::
localError(const MeasurementPoint& mp, const MeasurementError& me) const {
  const double
    phi(stripAngle(mp.x())), t(tan(phi)), c1(cos(phi)),
    cs(t/(1+t*t)), s2(t*cs), c2(1-s2), // rotation matrix

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

MeasurementError RadialStripTopology::
measurementError(const LocalPoint& p,  const LocalError& e) const {
  const double
    yHitToInter(yDistanceToIntersection(p.y())),
    t(yAxisOrientation() * p.x() / yHitToInter),   // tan(strip angle) 
    cs(t/(1+t*t)), s2(t*cs), c2(1-s2),             // rotation matrix

    T( angularWidth() * sqrt( pow(p.x(),2) + pow(yHitToInter,2)) ), // tangential measurement unit (local pitch)
    R( detHeight() / sqrt(c2) ),                                    // radial measurement unit (strip length) 

    uu(       ( c2*e.xx() - 2*cs*e.xy() + s2*e.yy() )   / (T*T) ),
    vv(       ( s2*e.xx() + 2*cs*e.xy() + c2*e.yy() )   / (R*R) ),
    uv( ( cs*( e.xx() - e.yy() ) + e.xy()*( c2 - s2 ) ) / (T*R) );
  
  return MeasurementError(uu, uv, vv);
}
 
float RadialStripTopology::pitch() const {  throw Genexception("RadialStripTopology::pitch() called - makes no sense, use localPitch(.) instead."); return 0.;}
float RadialStripTopology::
localPitch(const LocalPoint& lp) const {  // The local pitch is the local x width of the strip at the local (x,y)
  const int istrip = std::min(nstrips(), static_cast<int>(strip(lp)) + 1); // which strip number
  const float fangle = stripAngle(static_cast<float>(istrip) - 0.5); // angle of strip centre
  return
    yDistanceToIntersection( lp.y() ) * sin(angularWidth()) /
    ( cos(fangle-0.5*angularWidth()) * cos(fangle+0.5*angularWidth()) );
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
