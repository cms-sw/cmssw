#ifndef _RADIAL_STRIP_TOPOLOGY_H_
#define _RADIAL_STRIP_TOPOLOGY_H_

#include "Geometry/CommonTopologies/interface/StripTopology.h"

/**
 * \class RadialStripTopology
 * A StripTopology in which the component strips subtend a constant
 * angular width, and, if projected, intersect at a point.
 *
 * \author Tim Cox
 *
 * WARNING! Wherever 'float strip' is used the units of 'strip' are angular
 * widths of each strip. The range is from 0.0 at the extreme edge of the
 * 'first' strip at one edge of the detector, to nstrip*angular width
 * at the other edge. <BR>
 * The centre of the first strip is at strip = 0.5 <BR>
 * The centre of the last strip is at strip = 0.5 + (nstrip-1) <BR>
 * This is for consistency with CommonDet usage of 'float strip' (but
 * where units are strip pitch rather than strip angular width.)<BR>
 *
 * WARNING! If the mid-point along local y of the plane of strips does not correspond
 * to the local coordinate origin, set the final ctor argument appropriately. <BR>
 *
 * now is an abstract class to allow different specialization for tracker and muon
 */

class RadialStripTopology : public StripTopology {
 public:

 
  /** 
   * Destructor
   */
  virtual ~RadialStripTopology(){}

  // =========================================================
  // StripTopology interface - implement pure virtual methods
  // =========================================================

  /** 
   * LocalPoint on x axis for given 'strip'
   * 'strip' is a float in units of the strip (angular) width
   */
  virtual LocalPoint localPosition(float strip) const=0;

  /** 
   * LocalPoint for a given MeasurementPoint <BR>
   * What's a MeasurementPoint?  <BR>
   * In analogy with that used with TrapezoidalStripTopology objects,
   * a MeasurementPoint is a 2-dim object.<BR>
   * The first dimension measures the
   * angular position wrt central line of symmetry of detector,
   * in units of strip (angular) widths (range 0 to total angle subtended
   * by a detector).<BR>
   * The second dimension measures
   * the fractional position along the strip (range -0.5 to +0.5).<BR>
   * BEWARE! The components are not Cartesian.<BR>
   */
  virtual LocalPoint localPosition(const MeasurementPoint&) const=0;

  /** 
   * LocalError for a pure strip measurement, where 'strip'
   * is the (float) position (a 'phi' angle wrt y axis) and
   * stripErr2 is the sigma-squared. Both quantities are expressed in
   * units of theAngularWidth of a strip.
   */
  virtual LocalError localError(float strip, float stripErr2) const=0;

  /** 
   * LocalError for a given MeasurementPoint with known MeasurementError.
   * This may be used in Kalman filtering and hence must allow possible
   * correlations between the components.
   */
  virtual LocalError localError(const MeasurementPoint&, const MeasurementError&) const=0;

  /** 
   * Strip in which a given LocalPoint lies. This is a float which
   * represents the fractional strip position within the detector.<BR>
   * Returns zero if the LocalPoint falls at the extreme low edge of the
   * detector or BELOW, and float(nstrips) if it falls at the extreme high
   * edge or ABOVE.
   */
  virtual float strip(const LocalPoint&) const=0;


  /** 
   * BEWARE: calling pitch() throws an exception.<BR>
   * Pitch is conventional name for width of something, but this is
   * not sensible for a RadialStripTopology since strip widths vary with local y.
   * Use localPitch(.) instead.
   */
  virtual float pitch() const GCC11_FINAL;

  /** 
   * Pitch (strip width) at a given LocalPoint. <BR>
   * BEWARE: are you sure you really want to call this for a RadialStripTopology?
   */
  virtual float localPitch(const LocalPoint&) const=0;

  /** 
   * Angle between strip and symmetry axis (=local y axis)
   * for given strip. <BR>
   * This is like a phi angle but measured clockwise from y axis 
   * rather than counter clockwise from x axis.
   * Note that 'strip' is a float with a continuous range from 0 to 
   * float(nstrips) to cover the whole detector, and the centres of
   * strips correspond to half-integer values 0.5, 1.5, ..., nstrips-0.5
   * whereas values 1, 2, ... nstrips correspond to the upper phi edges of
   * the strips.
   */
  virtual float stripAngle(float strip) const=0;

  /** 
   * Total number of strips 
   */
  virtual int nstrips() const=0;

  /** 
   * Height of detector (= length of long symmetry axis of the plane of strips).
   */
  virtual float stripLength() const=0;

  /** 
   * Length of a strip passing through a given LocalPpoint
   */
  virtual float localStripLength(const LocalPoint& ) const=0;


  // =========================================================
  // Topology interface (not already implemented for 
  // StripTopology interface)
  // =========================================================

  virtual MeasurementPoint measurementPosition( const LocalPoint& ) const=0;

  virtual MeasurementError measurementError( const LocalPoint&, const LocalError& ) const=0;

  /** 
   * Channel number corresponding to a given LocalPoint.<BR>
   * This is effectively an integer version of strip(), with range 0 to
   * nstrips-1.  <BR>
   * LocalPoints outside the detector strip plane will be considered
   * as contributing to the edge channels 0 or nstrips-1.
   */
  virtual int channel( const LocalPoint& ) const=0;


  // =========================================================
  // RadialStripTopology interface itself
  // =========================================================

  /** 
   * Angular width of a each strip
   */
   virtual float angularWidth() const=0;

  /** 
   * Phi pitch of each strip (= angular width!)
   */
  virtual float phiPitch(void) const=0;

  /** 
   * Length of long symmetry axis of plane of strips
   */
   virtual float detHeight() const=0;

  /** 
   * y extent of strip plane
   */
   virtual float yExtentOfStripPlane() const=0; // same as detHeight()

  /** 
   * Distance from the intersection of the projections of
   * the extreme edges of the two extreme strips to the symmetry
   * centre of the plane of strips. 
   */
   virtual float centreToIntersection() const=0;
  /** 
   * (y) distance from intersection of the projections of the strips
   * to the local coordinate origin. Same as centreToIntersection()
   * if symmetry centre of strip plane coincides with local origin.
   */
   virtual float originToIntersection() const=0;
  /**
   * Convenience function to access azimuthal angle of extreme edge of first strip 
   * measured relative to long symmetry axis of the plane of strips. <BR>
   *
   * WARNING! This angle is measured clockwise from the local y axis 
   * which means it is in the conventional azimuthal phi plane, 
   * but azimuth is of course measured from local x axis not y. 
   * The range of this angle is
   *  -(full angle)/2 to +(full angle)/2. <BR>
   * where (full angle) = nstrips() * angularWidth(). <BR>
   *
   */
   virtual float phiOfOneEdge() const=0;

  /**
   * Local x where centre of strip intersects input local y <BR>
   * 'strip' should be in range 1 to nstrips() <BR>
   */
   virtual float xOfStrip(int strip, float y) const=0;
 
   /**
   * Nearest strip to given LocalPoint
   */
  virtual int nearestStrip(const LocalPoint&) const=0;

  /** 
   * y axis orientation, 1 means detector width increases with local y
   */
  virtual float yAxisOrientation() const=0;
  
  /**
   * Offset in local y between midpoint of detector (strip plane) extent and local origin
   */
  virtual float yCentreOfStripPlane() const=0;
  
  /**
   * Distance in local y from a hit to the point of intersection of projected strips
   */
  virtual float yDistanceToIntersection( float y ) const=0;

  friend std::ostream & operator<<(std::ostream&, const RadialStripTopology& );


};

#endif


