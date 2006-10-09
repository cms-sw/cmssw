// This is CSCLayerGeometry.cc

#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/src/OffsetRadialStripTopology.h>
#include <Geometry/CSCGeometry/src/ORedOffsetRST.h>
#include <Geometry/CSCGeometry/src/CSCWireGroupPackage.h>
#include <Geometry/Vector/interface/LocalPoint.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <CLHEP/Units/SystemOfUnits.h>

#include <algorithm>
#include <iostream>
#include <cmath>

CSCLayerGeometry::CSCLayerGeometry( int iChamberType,
         const TrapezoidalPlaneBounds& bounds,
         int nstrips, float stripOffset, float stripPhiPitch,
         const CSCWireGroupPackage& wg, float wireAngleInDegrees,
	 float ctiOffset)
  :   TrapezoidalPlaneBounds( bounds ), theWireTopology( 0 ),
      theStripTopology( 0 ), myName( "CSCLayerGeometry" ), 
      chamberType( iChamberType ) {

  LogDebug("CSC") << ": being constructed, this=" << this << "\n";

  // The TPB doesn't provide direct access to the half-length values
  // we've always been coding against (since that is what GEANT3 used!)
  //@@ For now, retrieve the half-lengths, Later perhaps make use
  //@@ of TPB's accessors directly?
  apothem     = bounds.length() / 2.;  
  hTopEdge    = bounds.width() / 2.;
  hBottomEdge = bounds.widthAtHalfLength() - hTopEdge; // t+b=2w

  // Ganged strips in ME1A?

  bool gangedME1A = ( iChamberType == 1 &&
		      CSCChamberSpecs::gangedStrips() );

  // Calculate 'whereStripsMeet' = distance from centre of chamber to
  // intersection point of projected strips.

  // In the perfect geometry this is the z-axis (for most rings of chambers)
  // since the trapezoids are truncated sectors of circles.


  // Radial or Trapezoidal strips? Only radial is realistic

  //  if ( CSCChamberSpecs::radialStrips() ) {

  // RST more tricky than TST, since we need to enforce the constraint
  // that the subtended angle is exactly the no. of strips * ang width
  // and that wasn't considered when setting the gas volume dimensions in
  // the input geometry. Best I can do, I think, is require half width
  // of layer along local x axis is (T+B)/2. 
  // Then tan(angle/2)=wid/w, and so w is:
     whereStripsMeet =
        0.5*(hTopEdge+hBottomEdge) / tan(0.5*nstrips*stripPhiPitch)
  // Add in the backed-out offset
       + ctiOffset;

     AbsOffsetRadialStripTopology* aStripTopology = 
        new OffsetRadialStripTopology(nstrips, stripPhiPitch,
	    2.*apothem, whereStripsMeet, stripOffset );

     if ( gangedME1A ) {
       theStripTopology = new ORedOffsetRST( *aStripTopology, 16 );
       delete aStripTopology;
     }
     else {
       theStripTopology = aStripTopology;
     }

  //  }


    //@@ HOW TO SET yOfFirstWire ? It should be explicit in the DDD.
    //@@ (To retrieve backward-compatibility I need a calculated value
    //@@ but I should calculate those values and write them in the DDD.) 

  double yOfFirstWire = 0;

  if ( CSCChamberSpecs::realWireGeometry() ) {
    //@@ For now just hack it in CSCWireTopology ctor.
    //@@ Passing iChamberType as yOfFirstFire is a terrible hack: 
    //@@ but need to know we have ME1A or ME11 inside the MEW constructor...
    yOfFirstWire = static_cast<float>( iChamberType );
  }
  else {
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
} 

CSCLayerGeometry::CSCLayerGeometry(const CSCLayerGeometry& melg) :
  TrapezoidalPlaneBounds(melg.hBottomEdge, melg.hTopEdge, melg.apothem,
			 0.5 * melg.thickness() ),
  theWireTopology(0), theStripTopology(0), 
  whereStripsMeet(melg.whereStripsMeet),
  hBottomEdge(melg.hBottomEdge), hTopEdge(melg.hTopEdge),
  apothem(melg.apothem)
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

    whereStripsMeet = melg.whereStripsMeet;
    hBottomEdge     = melg.hBottomEdge;
    hTopEdge        = melg.hTopEdge;
    apothem         = melg.apothem;
  }
  return *this;
}

CSCLayerGeometry::~CSCLayerGeometry()
{
  LogDebug("CSC") << ": being destroyed, this=" << this << 
    "\nDeleting theStripTopology=" << theStripTopology << 
    " and theWireTopology=" << theWireTopology << "\n";
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
  // But x involves mixing wire geometry with TrapezoidalPlaneBounds.

  // If the wires are NOT tilted, default to simple calculation...
  if ( fabs(wireAngle() ) < 1.E-6 )  {
    float y = yOfWireGroup( wireGroup );
    return LocalPoint( 0., y );
  }
  else {
    // w is "wire" at the center of the wire group
    float w = middleWireOfGroup( wireGroup );
    std::vector<float> store = wireValues( w );
    return LocalPoint( store[0], store[1] );
  }
}

float CSCLayerGeometry::lengthOfWireGroup( int wireGroup ) const {
  // Return length of 'wire' in the middle of the wire group
   float w = middleWireOfGroup( wireGroup );
   std::vector<float> store = wireValues( w );
   return store[2];
}
    
void CSCLayerGeometry::setTopology( CSCStripTopology * newTopology ) {
   delete theStripTopology;
   theStripTopology = newTopology;
}


LocalPoint CSCLayerGeometry::intersection( float m1, float c1, 
     float m2, float c2 ) const {

  // Calculate the point of intersection of two straight lines (in 2-dim)
  // BEWARE! Do not call with m1 = m2 ! No trapping !

  float x = (c2-c1)/(m1-m2);
  float y = (m1*c2-m2*c1)/(m1-m2);
  return LocalPoint( x, y );
}

std::vector<float> CSCLayerGeometry::wireValues( float wire ) const {
  // return x and y of mid-point of wire, and length of wire, as 3-dim vector.
  // If wire does not intersect active area the returned vector if filled with 0's.
  // ME11 is a special case so active area is effectively extended to be entire ME11
  // for either ME1a or ME1b active areas.

  std::vector<float> buf(3); // note all elem init to 0

  const float fprec = 1.E-06;

  // slope of wire
  float wangle = wireAngle();

  float mw = 0;
  if ( fabs(wangle) > fprec ) mw = tan( wireAngle() );

 // intercept of wire
  float cw = yOfWire( wire );


  LogDebug("CSC") << ": chamber type= " << chamberType << ", wire=" << wire << 
    ", wire angle = " << wangle << 
    ", intercept on y axis=" << cw << "\n";


  // slope & intercept of line defining one non-parallel side of chamber 
  float m1 = -2.*apothem/(hTopEdge-hBottomEdge);
  float c1 = -apothem*(hTopEdge+hBottomEdge)/(hTopEdge-hBottomEdge);

  // slope & intercept of other non-parallel side of chamber
  float m2 = -m1;
  float c2 =  c1;

  // wire intersects side 1 at
  LocalPoint pw1 = intersection(mw, cw, m1, c1);
  // wire intersects side 2 at
  LocalPoint pw2 = intersection(mw, cw, m2, c2);

  float x1 = pw1.x();
  float y1 = pw1.y();

  float x2 = pw2.x();
  float y2 = pw2.y();

  LogDebug("CSC") << ": wire intersects sides at " << 
                 "\n  x1=" << x1 << " y1=" << y1 << 
		   " x2=" << x2 << " y2=" << y2 << "\n";

  // WIRES ARE NOT TILTED?

  if ( fabs(wangle) < fprec ) {

    buf[0] = 0.;
    buf[1] = cw;
    buf[2] = sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );

    LogDebug("CSC") << ": wires are not tilted " << 
      "\n  mid-point: x=0 y=" << cw << ", length=" << buf[2] << "\n";

    return buf;    
  }

  // WIRES ARE TILTED

  // ME1a, ME1b basic geometry...half-lengths of top, bottom edges when
  // ME11 top width = 487.1 mm, bottom width = 201.3 mm, height = 1505 mm
  // ME1a height = 440 mm
  // ME1b height = 1065 mm
  const float lenOfme1b = 106.5;
  const float lenOfme1a = 44.0;
  const float htOfme1b = 48.71/2.;
  const float hbOfme1a = 20.13/2.;

  // ht and hb will be used to check where wire intersects chamber face
  float ht = hTopEdge;
  float hb = hBottomEdge;
  float mt = 0.; // slope of top edge
  float mb = 0.; //slope of bottom edge
  float ct = apothem;  // intercept top edge of chamber
  float cb = -apothem; // intercept bottom edge of chamber

  if (chamberType == 1 ) {
    ht = htOfme1b; // Active area is ME1a, but use top edge of ME11 i.e. ME1b
    ct += lenOfme1b; 
  }
  else if ( chamberType == 2 ) {
    hb = hbOfme1a; // Active are is ME1b, but use bottom edge of ME11 i.e. ME1a
    cb -= lenOfme1a;
  }
  
  LogDebug("CSC") << ": slopes & intercepts " <<
    "\n  mt=" << mt << " ct=" << ct << " mb=" << mb << " cb=" << cb <<
    "\n  m1=" << m1 << " c1=" << c1 << " m2=" << m2 << " c2=" << c2 << 
    "\n  mw=" << mw << " cw=" << cw << "\n";

  
  // wire intersects top side at
  LocalPoint pwt = intersection(mw, cw, mt, ct);
  // wire intersects bottom side at
  LocalPoint pwb = intersection(mw, cw, mb, cb);

  // get the local coordinates
  float xt = pwt.x();
  float yt = pwt.y();

  float xb = pwb.x();
  float yb = pwb.y();

  LogDebug("CSC") << ": wire intersects top & bottom at " << 
    "\n  xt=" << xt << " yt=" << yt << 
    " xb=" << xb << " yb=" << yb << "\n";

  float xWireEnd[4], yWireEnd[4];

  int i = 0;
  if ( fabs(x1) >= hb && fabs(x1) <= ht ) {
    // wire does intersect side edge 1 of chamber
    xWireEnd[i] = x1;
    yWireEnd[i] = y1;
    i++;
  }
  if ( fabs(xb) <= hb ) {
    // wire does intersect bottom edge of chamber
    xWireEnd[i] = xb;
    yWireEnd[i] = yb;
    i++;
  }
  if ( fabs(x2) >= hb && fabs(x2) <= ht ) {
    // wire does intersect side edge 2 of chamber
    xWireEnd[i] = x2;
    yWireEnd[i] = y2;
    i++;
  }
  if ( fabs(xt) <= ht ) {
    // wire does intersect top edge of chamber
    xWireEnd[i] = xt;
    yWireEnd[i] = yt;
    i++;
  }

  if ( i != 2 ) {
    // the wire does not intersect the (extended) active area

    LogDebug("CSC") << ": does not intersect active area \n";     
    //     throw cms::Exception("BadCSCGeometry") << "the wire has " << i <<
    //       " ends!" << "\n";

    return buf; // each elem is zero
  }
  
  LogDebug("CSC") << ": ME11 wire ends " << "\n";
  for ( int j = 0; j<i; j++ ) {
    LogDebug("CSC") << "  x = " << xWireEnd[j] << " y = " << yWireEnd[j] << "\n";
   }
 
  float d2 = (xWireEnd[0]-xWireEnd[1]) * (xWireEnd[0]-xWireEnd[1]) +
             (yWireEnd[0]-yWireEnd[1]) * (yWireEnd[0]-yWireEnd[1]);

  buf[0] = (xWireEnd[0]+xWireEnd[1])/2. ;
  buf[1] = (yWireEnd[0]+yWireEnd[1])/2. ;
  buf[2] = sqrt(d2) ;
  return buf;
}

std::ostream & operator<<(std::ostream & stream, const CSCLayerGeometry & lg) {
  stream << "LayerGeometry " << std::endl
         << "------------- " << std::endl
         << "numberOfStrips        " << lg.numberOfStrips() << std::endl
         << "numberOfWires         " << lg.numberOfWires() << std::endl
         << "numberOfWireGroups    " << lg.numberOfWireGroups() << std::endl
         << "wireAngle  (rad)      " << lg.wireAngle() << std::endl
    //         << "wireAngle  (deg)      " << lg.theWireAngle << std::endl
    //         << "sin(wireAngle)        " << lg.theWireSin << std::endl
    //         << "cos(wireAngle)        " << lg.theWireCos << std::endl
         << "wirePitch             " << lg.wirePitch() << std::endl
         << "stripPitch            " << lg.stripPitch() << std::endl
    //         << "numberOfWiresPerGroup " << lg.theNumberOfWiresPerGroup << std::endl
    //         << "numberOfWiresInLastGroup " << lg.theNumberOfWiresInLastGroup << std::endl
    //         << "wireOffset            " << lg.theWireOffset << std::endl
         << "hBottomEdge           " << lg.hBottomEdge << std::endl
         << "hTopEdge              " << lg.hTopEdge << std::endl
         << "apothem               " << lg.apothem << std::endl
         << "whereStripsMeet       " << lg.whereStripsMeet << std::endl;
    return stream;
}

