#ifndef Geometry_CSCGeometry_CSCLayerGeometry_H
#define Geometry_CSCGeometry_CSCLayerGeometry_H

/** \class CSCLayerGeometry
 *
 * Encapsulates the geometrical details of a CSCLayer
 * in a WireTopology for the wires and in a StripTopology for the strips.
 * Note that it does not have the capability of calculating
 * global values, so all values are in local coordinates.
 *
 * \author Tim Cox
 */

#include <Geometry/CSCGeometry/interface/CSCStripTopology.h>
#include <Geometry/CSCGeometry/interface/CSCWireTopology.h>
#include <DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h>
#include <DataFormats/GeometryVector/interface/LocalPoint.h>
#include <DataFormats/GeometrySurface/interface/LocalError.h>

#include <utility> // for std::pair

class CSCGeometry;
class CSCWireGroupPackage;

class CSCLayerGeometry : public TrapezoidalPlaneBounds { 

public:

 /**
   * Ctor from basic trapezoidal Chamber parameters.
   * \param geom The pointer to the actual CSCGeometry we're building.
   * \param iChamberType The index 1-9 for station/ring combination.
   * \param TrapezoidalPlaneBounds describing geometry of face.
   * \param nstrips No. of strips in cathode plane of a Layer.
   * \param stripOffset Alternate strip planes are relatively shifted by +/-0.25 strip widths.
   * \param stripPhiPitch Delta-phi width of strips (they're fan-shaped) in radians
   * \param whereStripsMeet radial distance from projected intersection of strips to centre of strip plane
   * \param extentOfStripPlane height of strip plane (along its long symmetry axis)
   * \param yCentreOfStripPlane local y of symmetry centre of strip plane (before any offset rotation)
   * \param wg CSCWireGroupPackage encapsulating wire group info.
   * \param wireAngleInDegrees angle of wires w.r.t local x axis.
   * \param yOfFirstWire local y coordinate of first (lowest) wire in wire plane - nearest narrow edge.
   * \param hThickness half-thickness of chamber layer in cm (i.e. half the gas gap).
   */
  CSCLayerGeometry( const CSCGeometry* geom, int iChamberType,
             const TrapezoidalPlaneBounds& bounds,
             int nstrips, float stripOffset, float stripPhiPitch, 
	     float whereStripsMeet, float extentOfStripPlane, float yCentreOfStripPlane,
	     const CSCWireGroupPackage& wg, float wireAngleInDegrees, double yOfFirstWire, float hThickness );

  CSCLayerGeometry(const CSCLayerGeometry& );

  CSCLayerGeometry& operator=( const CSCLayerGeometry& );

  virtual ~CSCLayerGeometry();

  /**
   * How many strips in layer
   */
  int numberOfStrips() const { 
    return theStripTopology->nstrips(); }
     
  /**
   * How many wires in layer
   */               
  int numberOfWires() const { 
    return theWireTopology->numberOfWires(); }

  /**
   * How many wire groups in Layer
   */
  int numberOfWireGroups() const {
    return theWireTopology->numberOfWireGroups(); }

  /**
   * How many wires in a wiregroup
   */
  int numberOfWiresPerGroup( int wireGroup ) const {
    return theWireTopology->numberOfWiresPerGroup( wireGroup ); }

  /**
   * Local point at which strip and wire intersect
   */
  LocalPoint stripWireIntersection( int strip, float wire) const;

  /**
   * Local point at which strip and centre of wire group intersect
   */
  LocalPoint stripWireGroupIntersection( int strip, int wireGroup) const;
  
  /**
   * Strip nearest a given local point
   */
  int nearestStrip(const LocalPoint & lp) const {
    return theStripTopology->nearestStrip(lp);
  }

  /**
   * Wire nearest a given local point
   */
  int nearestWire(const LocalPoint & lp) const {
    return theWireTopology->nearestWire( lp ); }

  /**
   * Wire group containing a given wire
   */
  int wireGroup(int wire) const {
    return theWireTopology->wireGroup( wire );
  }

  /** 
   * Electronics channel corresponding to a given strip
   * ...sometimes there will be more than one strip OR'ed into a channel
   */
  int channel(int strip)  const {
    return theStripTopology->channel(strip);
  }

  /**
   * Offset of strips from symmetrical distribution about local y axis
   * as a fraction of a strip (0 default, but usually +0.25 or -0.25)
   */
   float stripOffset( void ) const {return theStripTopology->stripOffset();}

  /**
   * Return +1 or -1 for a stripOffset of +0.25 or -0.25 respectively.
   * Requested by trigger people.
   */
   int stagger() const { return static_cast<int>( 4.1*stripOffset() ); }

  /**
   * The angle (in radians) of a strip wrt local x-axis.
   */  
  float stripAngle(int strip) const;

  /**
   * The angle (in radians) of (any) wire wrt local x-axis.
   */  
  float wireAngle() const {
    return theWireTopology->wireAngle(); }
  
  /**
   * The distance (in cm) between anode wires
   */
  float wirePitch() const {
    return theWireTopology->wirePitch(); }

  /**
   * The measurement resolution from wire groups (in cm.)
   * This approximates the measurement resolution in the local
   * y direction but may be too small by a factor of up to 1.26
   * due to stripAngle contributions which are neglected here.
   * The last wiregroup may have more wires than others.
   * The other wiregroups, including the first, are the same.
   * One day the wiregroups will be matched to the hardware
   * by using the DDD.
   */
 
  float yResolution( int wireGroup = 1 ) const {
    return theWireTopology->yResolution( wireGroup ); }

  /**
   * The phi width of the strips (radians)
   */
  float stripPhiPitch() const { 
    return theStripTopology->phiPitch(); }

  /**
   * The width of the strips (in middle)
   */
  float stripPitch() const { 
    //    return theStripTopology->pitch(); }
    return stripPitch( LocalPoint(0.,0.) ); }

  /**
   * The width of the strip at a given local point
   */
  float stripPitch(const LocalPoint & lp) const { 
    return theStripTopology->localPitch(lp); }
  
  /**
   *  The local x-position of the center of the strip.
   */
  float xOfStrip(int strip, float y=0.) const { 
    return theStripTopology->xOfStrip(strip, y); }

  /** Strip in which a given LocalPoint lies. This is a float which
   * represents the fractional strip position within the detector.<BR>
   * Returns zero if the LocalPoint falls at the extreme low edge of the
   * detector or BELOW, and float(nstrips) if it falls at the extreme high
   * edge or ABOVE.
   */
  float strip(const LocalPoint& lp) const { return theStripTopology->strip(lp); }

  /**
   * Middle of wire-group.
   * This is the central wire no. for a group with an odd no. of wires.
   * This is a pseudo-wire no. for a group with an even no. of wires.
   * Accordingly, it is non-integer.
   */
  float middleWireOfGroup( int wireGroup ) const {
    return theWireTopology->middleWireOfGroup( wireGroup ); }

  /**
   * Local y of a given wire 'number' (float) at given x
   */
  float yOfWire(float wire, float x=0.) const {
    return theWireTopology->yOfWire( wire, x ); }

  /**
   * Local y of a given wire group at given x
   */
  float yOfWireGroup(int wireGroup, float x=0.) const {
    return theWireTopology->yOfWireGroup( wireGroup, x ); }

  /**
   * Local coordinates of center of a wire group.
   * \WARNING Used to be centerOfWireGroup in ORCA
   * but that version now returns GlobalPoint.
   */
  LocalPoint localCenterOfWireGroup( int wireGroup ) const;

  /** 
   * Length of a wire group (center wire, across chamber face)
   */
  float lengthOfWireGroup( int wireGroup ) const;

  /**
   * Local y limits of the strip plane
   */
  std::pair<float, float> yLimitsOfStripPlane() const {
    return theStripTopology->yLimitsOfStripPlane();
  }

  /**
   * Is a supplied LocalPoint inside the strip region?
   * 
   * This is a more reliable fiducial cut for CSCs than the 'Bounds' of the GeomDet(Unit)
   * since those ranges are looser than the sensitive gas region.
   * There are three versions, to parallel those of the TrapezoidalPlaneBounds which
   * a naive user might otherwise employ.
   */
  bool inside( const Local3DPoint&, const LocalError&, float scale=1.f ) const;
  bool inside( const Local3DPoint& ) const;
  bool inside( const Local2DPoint& ) const;

  /**
   * Return estimate of the 2-dim point of intersection of a strip and a cluster of wires.
   *
   * Input arguments: a (float) strip number, and the wires which delimit a cluster of wires.
   * The wires are expected to be real wire numbers, and not wire-group numbers.<br>
   *
   * Returned: pair, with members: <br>
   * first: LocalPoint which is midway along "the" strip between the wire limits, <br>
   * or the chamber edges, as appropriate. <bf>
   * second: length of the strip between the wires (or edges as appropriate).
   */
  std::pair<LocalPoint, float> possibleRecHitPosition( float s, int w1, int w2 ) const;
      
  /**
   * Return 2-dim point at which a strip and a wire intersects.
   *
   * Input arguments: a (float) strip number, and an (int) wire. <br>
   * Output: LocalPoint which is at their intersection, or at extreme y
   * of wire plane, as appropriate. (If y is adjusted, x is also adjusted to keep it on same strip.)
   */
  LocalPoint intersectionOfStripAndWire( float s, int w) const;
	
  /**
   * Return the point of intersection of two straight lines (in 2-dim).
   *
   * Input arguments are pair(m1,c1) and pair(m2,c2) where m=slope, c=intercept (y=mx+c). <br>
   * BEWARE! Do not call with m1 = m2 ! No trapping !
   */
  LocalPoint intersectionOfTwoLines( std::pair<float, float> p1, std::pair<float, float> p2 ) const;

  /**
   * Transform strip and wire errors to local x, y frame.
   * Need to supply (central) strip of the hit.
   * The sigma's are in distance units.
   */
  LocalError localError( int strip, float sigmaStrip, float sigmaWire ) const;
  
  /**
   * 'The' Topology (i.e. Strip Topology) owned by this MELG
   */
  const CSCStripTopology* topology() const {
    return theStripTopology; 
  }

  /**
   *  This class takes ownership of the pointer, and will destroy it
   */
  void setTopology( CSCStripTopology * topology );

  /**
   * The Wire Topology owned by this MELG
   */
  const CSCWireTopology* wireTopology() const {
    return theWireTopology; 
  }

  /**
   * Utility method to handle proper copying of the class
   */
  virtual Bounds* clone() const { 
    return new CSCLayerGeometry(*this);
  }

  /**
   * Output operator for members of class.
   */
  friend std::ostream & operator<<(std::ostream &, const CSCLayerGeometry &);

private:

  // The wire information is encapsulated in a CSCWireTopology
  // This class is passed the pointer and takes ownership...
  // e.g. it destroys it.
  
  CSCWireTopology* theWireTopology;

  // The strip information is encapsulated in a CSCStripTopology
  // This class owns the pointer... so that copying works with
  // derived classes, need a clone() method which can be virtual.

  CSCStripTopology* theStripTopology;

  // Cache the trapezoid dimensions even though they could
  // be accessed from the TrapezoidalPlaneBounds

  float hBottomEdge;
  float hTopEdge;
  float apothem;

  const std::string myName;
  int chamberType;
};
#endif
