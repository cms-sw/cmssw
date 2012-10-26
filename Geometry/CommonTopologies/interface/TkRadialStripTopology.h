#ifndef _TkRADIAL_STRIP_TOPOLOGY_H_
#define _TkRADIAL_STRIP_TOPOLOGY_H_

#include "Geometry/CommonTopologies/interface/RadialStripTopology.h"

/**
 * \class TkRadialStripTopology
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
 * this version is optimized for tracker and is FINAL
 */

class TkRadialStripTopology GCC11_FINAL : public RadialStripTopology {
 public:

  /** 
   * Constructor from:
   *    \param ns number of strips
   *    \param aw angular width of a strip
   *    \param dh detector height (usually 2 x apothem of TrapezoidalPlaneBounds)
   *    \param r radial distance from symmetry centre of detector to the point at which 
   *    the outer edges of the two extreme strips (projected) intersect.
   *    \param yAx orientation of local y axis: 1 means pointing from the smaller side of
   *    the module to the larger side (along apothem), and -1 means in the 
   *    opposite direction, i.e. from the larger side along the apothem to the 
   *    smaller side. Default value is 1. 
   *    \param yMid local y offset if mid-point of detector (strip plane) does not coincide with local origin.
   *    This decouples the extent of strip plane from the boundary of the detector in which the RST is embedded.
   */
  TkRadialStripTopology( int ns, float aw, float dh, float r, int yAx = 1, float yMid = 0.);

  /** 
   * Destructor
   */
  ~TkRadialStripTopology(){}

  // =========================================================
  // StripTopology interface - implement pure methods
  // =========================================================

  /** 
   * LocalPoint on x axis for given 'strip'
   * 'strip' is a float in units of the strip (angular) width
   */
  LocalPoint localPosition(float strip) const;

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
  LocalPoint localPosition(const MeasurementPoint&) const;

  /** 
   * LocalError for a pure strip measurement, where 'strip'
   * is the (float) position (a 'phi' angle wrt y axis) and
   * stripErr2 is the sigma-squared. Both quantities are expressed in
   * units of theAngularWidth of a strip.
   */
  LocalError localError(float strip, float stripErr2) const;

  /** 
   * LocalError for a given MeasurementPoint with known MeasurementError.
   * This may be used in Kalman filtering and hence must allow possible
   * correlations between the components.
   */
  LocalError localError(const MeasurementPoint&, const MeasurementError&) const;

  /** 
   * Strip in which a given LocalPoint lies. This is a float which
   * represents the fractional strip position within the detector.<BR>
   * Returns zero if the LocalPoint falls at the extreme low edge of the
   * detector or BELOW, and float(nstrips) if it falls at the extreme high
   * edge or ABOVE.
   */
  float strip(const LocalPoint&) const;

  // the number of strip span by the segment between the two points..
  float coveredStrips(const LocalPoint& lp1, const LocalPoint& lp2)  const ; 


  /** 
   * Pitch (strip width) at a given LocalPoint. <BR>
   * BEWARE: are you sure you really want to call this for a RadialStripTopology?
   */
  float localPitch(const LocalPoint&) const;

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
  float stripAngle(float strip) const;

  /** 
   * Total number of strips 
   */
  int nstrips() const { return theNumberOfStrips; }

  /** 
   * Height of detector (= length of long symmetry axis of the plane of strips).
   */
  float stripLength() const { return theDetHeight; }

  /** 
   * Length of a strip passing through a given LocalPpoint
   */
  float localStripLength(const LocalPoint& ) const;


  // =========================================================
  // Topology interface (not already implemented for 
  // StripTopology interface)
  // =========================================================

  MeasurementPoint measurementPosition( const LocalPoint& ) const;

  MeasurementError measurementError( const LocalPoint&, const LocalError& ) const;

  /** 
   * Channel number corresponding to a given LocalPoint.<BR>
   * This is effectively an integer version of strip(), with range 0 to
   * nstrips-1.  <BR>
   * LocalPoints outside the detector strip plane will be considered
   * as contributing to the edge channels 0 or nstrips-1.
   */
  int channel( const LocalPoint& ) const;


  // =========================================================
  // RadialStripTopology interface itself
  // =========================================================

  /** 
   * Angular width of a each strip
   */
  float angularWidth() const { return theAngularWidth;}

  /** 
   * Phi pitch of each strip (= angular width!)
   */
  float phiPitch(void) const { return angularWidth(); }

  /** 
   * Length of long symmetry axis of plane of strips
   */
  float detHeight() const { return theDetHeight;}

  /** 
   * y extent of strip plane
   */
  float yExtentOfStripPlane() const { return theDetHeight; } // same as detHeight()

  /** 
   * Distance from the intersection of the projections of
   * the extreme edges of the two extreme strips to the symmetry
   * centre of the plane of strips. 
   */
  float centreToIntersection() const { return theCentreToIntersection; }

  /** 
   * (y) distance from intersection of the projections of the strips
   * to the local coordinate origin. Same as centreToIntersection()
   * if symmetry centre of strip plane coincides with local origin.
   */
  float originToIntersection() const { return (theCentreToIntersection - yCentre); }

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
  float phiOfOneEdge() const { return thePhiOfOneEdge; }

  /**
   * Local x where centre of strip intersects input local y <BR>
   * 'strip' should be in range 1 to nstrips() <BR>
   */
  float xOfStrip(int strip, float y) const;
 
   /**
   * Nearest strip to given LocalPoint
   */
  int nearestStrip(const LocalPoint&) const;

  /** 
   * y axis orientation, 1 means detector width increases with local y
   */
  float yAxisOrientation() const { return theYAxisOrientation; }

  /**
   * Offset in local y between midpoint of detector (strip plane) extent and local origin
   */
  float yCentreOfStripPlane() const { return yCentre; }

  /**
   * Distance in local y from a hit to the point of intersection of projected strips
   */
  float yDistanceToIntersection( float y ) const;

 private:

  int   theNumberOfStrips; // total no. of strips in plane of strips
  float theAngularWidth;   // angle subtended by each strip = phi pitch
  float theAWidthInverse;   // inverse of above
  float theTanAW;          // its tangent  
  float theDetHeight;      // length of long symmetry axis = twice the apothem of the enclosing trapezoid
  float theCentreToIntersection;  // distance centre of detector face to intersection of edge strips (projected)
  float thePhiOfOneEdge;   // local 'phi' of one edge of plane of strips (I choose it negative!)
  float theTanOfOneEdge;   // the positive tangent of the above...
  float   theYAxisOrientation; // 1 means y axis going from smaller to larger side, -1 means opposite direction
  float yCentre; // Non-zero if offset in local y between midpoint of detector (strip plane) extent and local origin.
};

#endif


