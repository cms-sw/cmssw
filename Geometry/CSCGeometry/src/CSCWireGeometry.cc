#include "Geometry/CSCGeometry/interface/CSCWireGeometry.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <cmath>

LocalPoint CSCWireGeometry::intersection( float m1, float c1,
					  float m2, float c2 ) const {

  // Calculate the point of intersection of two straight lines (in 2-dim)
  // BEWARE! Do not call with m1 = m2 ! No trapping !

  float x = (c2-c1)/(m1-m2);
  float y = (m1*c2-m2*c1)/(m1-m2);
  return LocalPoint( x, y );
}

std::vector<float> CSCWireGeometry::wireValues( float wire ) const {

  // return x and y of mid-point of wire, and length of wire, as 3-dim vector.
  // If wire does not intersect active area the returned vector is filled with 0's.

  std::pair< LocalPoint, LocalPoint > ends = wireEnds( wire );

  std::vector<float> buf(3); // note all elem init to 0
  
  buf[0] = (ends.first.x() + ends.second.x())/2.; // x is first elem of first & second pairs
  buf[1] = (ends.first.y() + ends.second.y())/2.; // y is second elem of first & second pairs
  float d2 = (ends.first.x() - ends.second.x()) * (ends.first.x() - ends.second.x()) +
             (ends.first.y() - ends.second.y()) * (ends.first.y() - ends.second.y());
  buf[2] = sqrt(d2) ;
  return buf;
}

std::pair< LocalPoint, LocalPoint > CSCWireGeometry::wireEnds( float wire ) const {

  // return local (x, y) of each end of wire.
  // If wire does not intersect active area set all values to 0.
  
  const float fprec = 1.E-06;

  // slope of wire
  float wangle = wireAngle();
  float mw = 0;
  if ( fabs(wangle) > fprec ) mw = tan( wireAngle() );

  // intercept of wire
  float cw = yOfWire( wire );
  
  LogTrace("CSCWireGeometry|CSC") << "CSCWireGeometry: wire=" << wire <<
    ", wire angle = " << wangle <<
    ", intercept on y axis=" << cw;
  
  // Find extent of wire plane
  double ww = wideWidthOfPlane();
  double nw = narrowWidthOfPlane();
  double len = lengthOfPlane();
  
  // slope & intercept of line defining one non-parallel edge of wire-plane trapezoid
  float m1 = 2.*len/(ww-nw);
  float c1 = 0.;
  if ( fabs(wangle) < fprec ) {
    c1 = yOfFirstWire() - nw*len/(ww-nw) ; // non-ME11
  }
  else {
    c1 = -len/2. - nw*len/(ww-nw); // ME11
  }

  // slope & intercept of other non-parallel edge of wire-plane trapezoid
  float m2 = -m1;
  float c2 =  c1;

  // wire intersects edge 1 at
  LocalPoint pw1 = intersection(mw, cw, m1, c1);
  // wire intersects edge 2 at
  LocalPoint pw2 = intersection(mw, cw, m2, c2);

  float x1 = pw1.x();
  float y1 = pw1.y();

  float x2 = pw2.x();
  float y2 = pw2.y();

  LogTrace("CSCWireGeometry|CSC") << "CSCWireGeometry: wire intersects edges of plane at " <<
    "\n  x1=" << x1 << " y1=" << y1 <<
    " x2=" << x2 << " y2=" << y2;
  
  // WIRES ARE NOT TILTED?

  if ( fabs(wangle) < fprec ) {

    LogTrace("CSCWireGeometry|CSC") << "CSCWireGeometry: wires are not tilted ";
    return std::pair< LocalPoint, LocalPoint >( LocalPoint(x1,y1), LocalPoint(x2,y2) );  
  }
  
  // WIRES ARE TILTED

  // ht and hb will be used to check where wire intersects edges of wire plane
  float ht = ww/2. ;
  float hb = nw/2. ;
  float mt = 0.; // slope of top edge
  float mb = 0.; //slope of bottom edge
  float cb = -len/2.; // intercept bottom edge of wire plane
  float ct =  len/2.; // intercept top edge of wire plane
  
  LogTrace("CSCWireGeometry|CSC") <<  "CSCWireGeometry: slopes & intercepts " <<
    "\n  mt=" << mt << " ct=" << ct << " mb=" << mb << " cb=" << cb <<
    "\n  m1=" << m1 << " c1=" << c1 << " m2=" << m2 << " c2=" << c2 <<
    "\n  mw=" << mw << " cw=" << cw;
  
  // wire intersects top edge at
  LocalPoint pwt = intersection(mw, cw, mt, ct);
  // wire intersects bottom edge at
  LocalPoint pwb = intersection(mw, cw, mb, cb);
  
  // get the local coordinates
  float xt = pwt.x();
  float yt = pwt.y();
  
  float xb = pwb.x();
  float yb = pwb.y();
  
  LogTrace("CSCWireGeometry|CSC") << "CSCWireGeometry: wire intersects top & bottom of wire plane at " <<
    "\n  xt=" << xt << " yt=" << yt <<
    " xb=" << xb << " yb=" << yb ;
  
  float xWireEnd[4], yWireEnd[4];
  
  int i = 0;
  if ( fabs(x1) >= hb && fabs(x1) <= ht ) {
    // wire does intersect side edge 1 of wire plane
    xWireEnd[i] = x1;
    yWireEnd[i] = y1;
    i++;
  }
  if ( fabs(xb) <= hb ) {
    // wire does intersect bottom edge of wire plane
    xWireEnd[i] = xb;
    yWireEnd[i] = yb;
    i++;
  }
  if ( fabs(x2) >= hb && fabs(x2) <= ht ) {
    // wire does intersect side edge 2 of wire plane
    xWireEnd[i] = x2;
    yWireEnd[i] = y2;
    i++;
  }
  if ( fabs(xt) <= ht ) {
    // wire does intersect top edge of wire plane
    xWireEnd[i] = xt;
    yWireEnd[i] = yt;
    i++;
  }
  
  if ( i != 2 ) {
    // the wire does not intersect the wire plane (!)

    LogTrace("CSCWireGeometry|CSC") << "CSCWireGeometry: wire does not intersect wire plane!!";
    //     throw cms::Exception("BadCSCGeometry") << "the wire has " << i <<
    //       " ends!" << "\n";

    return std::pair< LocalPoint, LocalPoint >( LocalPoint(0.,0.), LocalPoint(0.,0.) ); 
  }
  
  LogTrace("CSCWireGeometry|CSC") << "CSCWireGeometry: ME11 wire ends ";
  for ( int j = 0; j<i; j++ ) {
    LogTrace("CSCWireGeometry|CSC") << "  x = " << xWireEnd[j] << " y = " << yWireEnd[j];
  }

  return std::pair< LocalPoint, LocalPoint >
   ( LocalPoint(xWireEnd[0],yWireEnd[0]), LocalPoint(xWireEnd[1],yWireEnd[1]) );  
}

//@@ COULD/SHOULD BE IMPLEMENTED IN Slanted & Nonslanted DERIVED CLASSES

std::pair<float, float> CSCWireGeometry::equationOfWire( float wire ) const {
  
  const float fprec = 1.E-06;

  // slope of wire
  float wangle = wireAngle(); 
  float mw = 0;
  if ( fabs(wangle) > fprec ) mw = tan( wangle );

  // intercept of wire
  float cw = yOfWire( wire );

  LogTrace("CSCWireGeometry|CSC") << "CSCWireGeometry: wire=" << wire <<
    ", wire angle = " << wangle <<
    ", intercept on y axis=" << cw;

  return std::pair<float,float>(mw, cw);
}

//@@ COULD/SHOULD BE IMPLEMENTED IN Slanted & Nonslanted DERIVED CLASSES

std::pair<float, float> CSCWireGeometry::yLimitsOfWirePlane() const{

  const float fprec = 0.1; // wire angle is either 0 or 29 degrees = 0.506 rads
  float ylow = yOfFirstWire(); // non-ME11 chambers
  float wangle = wireAngle();  
  if ( fabs(wangle) > fprec ) {
    ylow += tan( std::abs(wangle) ) * narrowWidthOfPlane()/2.; // correction for ME11
  }
  float yhigh = ylow + lengthOfPlane(); // add extent of wire plane in y
  
  return std::pair<float, float>(ylow, yhigh);
}

