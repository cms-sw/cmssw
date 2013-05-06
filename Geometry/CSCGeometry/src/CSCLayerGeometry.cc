// This is CSCLayerGeometry.cc

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>

#include <Geometry/CSCGeometry/src/CSCUngangedStripTopology.h>
#include <Geometry/CSCGeometry/src/CSCGangedStripTopology.h>
#include <Geometry/CSCGeometry/src/CSCWireGroupPackage.h>


#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <algorithm>
#include <iostream>
#include <cmath> // for M_PI_2 via math.h, as long as appropriate compiler flag switched on to allow acces

// Complicated initialization list since the chamber Bounds passed in must have its thickness reset to that of the layer
// Note for half-widths, t + b = 2w and the TPB only has accessors for t and w so b = 2w - t

CSCLayerGeometry::CSCLayerGeometry( const CSCGeometry* geom, int iChamberType,
         const TrapezoidalPlaneBounds& bounds,
         int nstrips, float stripOffset, float stripPhiPitch,
	 float whereStripsMeet, float extentOfStripPlane, float yCentreOfStripPlane,
	 const CSCWireGroupPackage& wg, float wireAngleInDegrees, double yOfFirstWire, float hThickness )
  :   TrapezoidalPlaneBounds( bounds.widthAtHalfLength() - bounds.width()/2., bounds.width()/2., bounds.length()/2., hThickness ), 
      theWireTopology( 0 ), theStripTopology( 0 ), 
      hBottomEdge( bounds.widthAtHalfLength() - bounds.width()/2. ), 
      hTopEdge( bounds.width()/2. ), apothem( bounds.length()/2. ),  
      myName( "CSCLayerGeometry" ), chamberType( iChamberType ) {

  LogTrace("CSCLayerGeometry|CSC") << myName <<": being constructed, this=" << this;

  // Ganged strips in ME1A?
  bool gangedME1A = ( iChamberType == 1 && geom->gangedStrips() );

  CSCStripTopology* aStripTopology = 
        new CSCUngangedStripTopology(nstrips, stripPhiPitch,
	    extentOfStripPlane, whereStripsMeet, stripOffset, yCentreOfStripPlane );

  if ( gangedME1A ) {
    theStripTopology = new CSCGangedStripTopology( *aStripTopology, 16 );
    delete aStripTopology;
  }
  else {
    theStripTopology = aStripTopology;
  }

  if ( ! geom->realWireGeometry() ) {
    // Approximate ORCA_8_8_0 and earlier calculated geometry...
    float wangler = wireAngleInDegrees*degree; // convert angle to radians
    float wireCos = cos(wangler);
    float wireSin = sin(wangler);
    float y2 = apothem * wireCos + hBottomEdge * fabs(wireSin);
    float wireSpacing = wg.wireSpacing/10.; // in cm
    float wireOffset = -y2 + wireSpacing/2.;
    yOfFirstWire = wireOffset/wireCos;
  }

  theWireTopology = new CSCWireTopology( wg, yOfFirstWire, wireAngleInDegrees );
  LogTrace("CSCLayerGeometry|CSC") << myName <<": constructed: "<< *this;
} 

CSCLayerGeometry::CSCLayerGeometry(const CSCLayerGeometry& melg) :
  TrapezoidalPlaneBounds(melg.hBottomEdge, melg.hTopEdge, melg.apothem,
			 0.5 * melg.thickness() ),
  theWireTopology(0), theStripTopology(0), 
  hBottomEdge(melg.hBottomEdge), hTopEdge(melg.hTopEdge),
  apothem(melg.apothem), chamberType(melg.chamberType) 
{
  // CSCStripTopology is abstract, so need clone()
  if (melg.theStripTopology) theStripTopology = melg.theStripTopology->clone();
  // CSCWireTopology is concrete, so direct copy
  if (melg.theWireTopology) theWireTopology = new CSCWireTopology(*(melg.theWireTopology));
}

CSCLayerGeometry& CSCLayerGeometry::operator=(const CSCLayerGeometry& melg)
{
  if ( &melg != this ) {
    delete theStripTopology;
    if ( melg.theStripTopology )
      theStripTopology=melg.theStripTopology->clone();
    else
      theStripTopology=0;

    delete theWireTopology;
    if ( melg.theWireTopology )
      theWireTopology=new CSCWireTopology(*(melg.theWireTopology));
    else
      theWireTopology=0;

    hBottomEdge     = melg.hBottomEdge;
    hTopEdge        = melg.hTopEdge;
    apothem         = melg.apothem;
  }
  return *this;
}

CSCLayerGeometry::~CSCLayerGeometry()
{
  LogTrace("CSCLayerGeometry|CSC") << myName << ": being destroyed, this=" << this << 
    "\nDeleting theStripTopology=" << theStripTopology << 
    " and theWireTopology=" << theWireTopology;
  delete theStripTopology;
  delete theWireTopology;
}


LocalPoint 
CSCLayerGeometry::stripWireIntersection( int strip, float wire ) const
{
  // This allows _float_ wire no. so that we can calculate the
  // intersection of a strip with the mid point of a wire group 
  // containing an even no. of wires (which is not an actual wire),
  // as well as for a group containing an odd no. of wires.

  // Equation of wire and strip as straight lines in local xy
  // y = mx + c where m = tan(angle w.r.t. x axis)
  // At the intersection x = -(cs-cw)/(ms-mw)
  // At y=0, 0 = ms * xOfStrip(strip) + cs => cs = -ms*xOfStrip
  // At x=0, yOfWire(wire) = 0 + cw => cw = yOfWire

  float ms = tan( stripAngle(strip) );
  float mw = tan( wireAngle() );
  float xs = xOfStrip(strip);
  float xi = ( ms * xs + yOfWire(wire) ) / ( ms - mw );
  float yi = ms * (xi - xs );

  return LocalPoint(xi, yi);
}

LocalPoint CSCLayerGeometry::stripWireGroupIntersection( int strip, int wireGroup) const
{
  // middleWire is only an actual wire for a group with an odd no. of wires
  float middleWire = middleWireOfGroup( wireGroup );
  return stripWireIntersection(strip, middleWire);
}

float CSCLayerGeometry::stripAngle(int strip) const 
{
  // Cleverly subtly change meaning of stripAngle once more.
  // In TrapezoidalStripTopology it is angle measured
  // counter-clockwise from y axis. 
  // In APTST and RST it is angle measured 
  // clockwise from y axis.
  // Output of this function is measured counter-clockwise 
  // from x-axis, so it is a conventional 2-dim azimuthal angle
  // in the (x,y) local coordinates

  // We want angle at centre of strip (strip N covers
  // *float* range N-1 to N-epsilon)

  return M_PI_2 - theStripTopology->stripAngle(strip-0.5);
}

LocalPoint CSCLayerGeometry::localCenterOfWireGroup( int wireGroup ) const {

  // It can use CSCWireTopology::yOfWireGroup for y,
  // But x requires mixing with 'extent' of wire plane

  // If the wires are NOT tilted, default to simple calculation...
  if ( fabs(wireAngle() ) < 1.E-6 )  {
    float y = yOfWireGroup( wireGroup );
    return LocalPoint( 0., y );
  }
  else {
    // w is "wire" at the center of the wire group
    float w = middleWireOfGroup( wireGroup );
    std::vector<float> store = theWireTopology->wireValues( w );
    return LocalPoint( store[0], store[1] );
  }
}

float CSCLayerGeometry::lengthOfWireGroup( int wireGroup ) const {
  // Return length of 'wire' in the middle of the wire group
   float w = middleWireOfGroup( wireGroup );
   std::vector<float> store = theWireTopology->wireValues( w );
   return store[2];
}

std::pair<LocalPoint, float> CSCLayerGeometry::possibleRecHitPosition( float s, int w1, int w2 ) const {
	
  LocalPoint sw1 = intersectionOfStripAndWire( s, w1 );
  LocalPoint sw2 = intersectionOfStripAndWire( s, w2 );
		
  // Average the two points
  LocalPoint midpt( (sw1.x()+sw2.x())/2., (sw1.y()+sw2.y())/2 );
	
  // Length of strip crossing this group of wires
  float length = sqrt( (sw1.x()-sw2.x())*(sw1.x()-sw2.x()) + 
                     (sw1.y()-sw2.y())*(sw1.y()-sw2.y()) );
	
  return std::pair<LocalPoint,float>( midpt, length );
}

LocalPoint CSCLayerGeometry::intersectionOfStripAndWire( float s, int w) const {
	
  std::pair<float, float> pw = theWireTopology->equationOfWire( static_cast<float>(w) );
  std::pair<float, float> ps = theStripTopology->equationOfStrip( s );
  LocalPoint sw = intersectionOfTwoLines( ps, pw );
	
  // If point falls outside wire plane, at extremes in local y, 
  // replace its y by that of appropriate edge of wire plane
  if ( !(theWireTopology->insideYOfWirePlane( sw.y() ) ) ) {
     float y  = theWireTopology->restrictToYOfWirePlane( sw.y() );
     // and adjust x to match new y
     float x = sw.x() + (y - sw.y())*tan(theStripTopology->stripAngle(s));
     sw = LocalPoint(x, y);
  }
	
  return sw;
}

LocalPoint CSCLayerGeometry::intersectionOfTwoLines( std::pair<float, float> p1, std::pair<float, float> p2 ) const {

  // Calculate the point of intersection of two straight lines (in 2-dim)
  // input arguments are pair(m1,c1) and pair(m2,c2) where m=slope, c=intercept (y=mx+c)
  // BEWARE! Do not call with m1 = m2 ! No trapping !

  float m1 = p1.first;
  float c1 = p1.second;
  float m2 = p2.first;
  float c2 = p2.second;
  float x = (c2-c1)/(m1-m2);
  float y = (m1*c2-m2*c1)/(m1-m2);
  return LocalPoint( x, y );
}


LocalError CSCLayerGeometry::localError( int strip, float sigmaStrip, float sigmaWire ) const {
  // Input sigmas are expected to be in _distance units_
  // - uncertainty in strip measurement (typically from Gatti fit, value is in local x units)
  // - uncertainty in wire measurement (along direction perpendicular to wires)

  float wangle   = this->wireAngle();
  float strangle = this->stripAngle( strip );

  float sinAngdif  = sin(strangle-wangle);
  float sinAngdif2 = sinAngdif * sinAngdif;
  
  float du = sigmaStrip/sin(strangle); // sigmaStrip is just x-component of strip error
  float dv = sigmaWire;

  // The notation is
  // wsins = wire resol  * sin(strip angle)
  // wcoss = wire resol  * cos(strip angle)
  // ssinw = strip resol * sin(wire angle)
  // scosw = strip resol * cos(wire angle)

  float wsins = dv * sin(strangle);
  float wcoss = dv * cos(strangle);
  float ssinw = du * sin(wangle);
  float scosw = du * cos(wangle);

  float dx2 = (scosw*scosw + wcoss*wcoss)/sinAngdif2;
  float dy2 = (ssinw*ssinw + wsins*wsins)/sinAngdif2;
  float dxy = (scosw*ssinw + wcoss*wsins)/sinAngdif2;
          
  return LocalError(dx2, dxy, dy2);
}

bool CSCLayerGeometry::inside( const Local3DPoint& lp ) const {
  bool result = false;
  const float epsilon = 1.e-06;
  if ( fabs( lp.z() ) < thickness()/2. ) { // thickness of TPB is that of gas layer
    std::pair<float, float> ylims = yLimitsOfStripPlane();
    if ( (lp.y() > ylims.first) && (lp.y() < ylims.second) ) {
      // 'strip' returns float value between 0. and float(Nstrips) and value outside
      // is set to 0. or float(Nstrips)... add a conservative precision of 'epsilon'
      if ( ( theStripTopology->strip(lp) > epsilon ) && 
           ( theStripTopology->strip(lp) < (numberOfStrips() - epsilon) ) ) result = true;
    }
  }
  return result;
}

bool CSCLayerGeometry::inside( const Local2DPoint& lp ) const {
  LocalPoint lp2( lp.x(), lp.y(), 0. );
  return inside( lp2 );
}

bool CSCLayerGeometry::inside( const Local3DPoint& lp, const LocalError& le, float scale ) const {
  // Effectively consider that the LocalError components extend the area which is acceptable.
  // Form a little box centered on the point, with x, y diameters defined by the errors
  // and require that ALL four corners of the box fall outside the strip region for failure

  // Note that LocalError is 2-dim x,y and doesn't supply a z error
  float deltaX = scale*sqrt(le.xx());
  float deltaY = scale*sqrt(le.yy());

  LocalPoint lp1( lp.x()-deltaX, lp.y()-deltaY, lp.z() );
  LocalPoint lp2( lp.x()-deltaX, lp.y()+deltaY, lp.z() );
  LocalPoint lp3( lp.x()+deltaX, lp.y()+deltaY, lp.z() );
  LocalPoint lp4( lp.x()+deltaX, lp.y()-deltaY, lp.z() );

  return ( inside(lp1) || inside(lp2) || inside(lp3) || inside(lp4) );
}

void CSCLayerGeometry::setTopology( CSCStripTopology * newTopology ) {
   delete theStripTopology;
   theStripTopology = newTopology;
}

std::ostream & operator<<(std::ostream & stream, const CSCLayerGeometry & lg) {
  stream << "LayerGeometry " << std::endl
         << "------------- " << std::endl
         << "numberOfStrips               " << lg.numberOfStrips() << std::endl
         << "numberOfWires                " << lg.numberOfWires() << std::endl
         << "numberOfWireGroups           " << lg.numberOfWireGroups() << std::endl
         << "wireAngle  (rad)             " << lg.wireAngle() << std::endl
    //         << "wireAngle  (deg)      " << lg.theWireAngle << std::endl
    //         << "sin(wireAngle)        " << lg.theWireSin << std::endl
    //         << "cos(wireAngle)        " << lg.theWireCos << std::endl
         << "wirePitch                    " << lg.wirePitch() << std::endl
         << "stripPitch                   " << lg.stripPitch() << std::endl
    //         << "numberOfWiresPerGroup " << lg.theNumberOfWiresPerGroup << std::endl
    //         << "numberOfWiresInLastGroup " << lg.theNumberOfWiresInLastGroup << std::endl
    //         << "wireOffset            " << lg.theWireOffset << std::endl
    //         << "whereStripsMeet       " << lg.whereStripsMeet << std::endl;
         << "hBottomEdge                  " << lg.hBottomEdge << std::endl
         << "hTopEdge                     " << lg.hTopEdge << std::endl
         << "apothem                      " << lg.apothem << std::endl
         << "length (should be 2xapothem) " << lg.length() << std::endl
         << "thickness                    " << lg.thickness() << std::endl;
    return stream;
}

