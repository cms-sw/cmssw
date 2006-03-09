#ifndef _RADIAL_STRIP_TOPOLOGY_H_
#define _RADIAL_STRIP_TOPOLOGY_H_

#include "Geometry/CSCGeometry/interface/CSCStripTopology.h"

/**
 * \class RadialStripTopology
 * A CSCStripTopology in which the component strips subtend a constant
 * angular width, and, if projected, intersect at a point.
 *
 * \author Tim Cox
 *
 * In the Endcap Muon CSCs, the cathode strips are not strictly trapezoidal 
 * but fan-shaped, each subtending a constant azimuthal angle, and project 
 * to a point. In every station and ring except for ME13
 * the nominal (perfect) geometry has this point of intersection on the
 * beam line. That constraint is unused as far as possible in order
 * to allow non-perfect geometry and misalignment scenarios.<BR>
 *
 * WARNING! Wherever 'float strip' is used the units of 'strip' are angular
 * widths of each strip. The range is from 0.0 at the extreme edge of the
 * 'first' strip at one edge of the detector, to nstrip*angular width
 * at the other edge. <BR>
 * The centre of the first strip is at strip = 0.5 <BR>
 * The centre of the last strip is at strip = 0.5 * nstrip <BR>
 * This is for consistency with CommonDet usage of 'float strip' (but
 * where units are strip pitch rather than strip angular width.)<BR>
 *
 * WARNING! We measure angles from the local y axis which means they are in
 * the conventional azimuthal phi plane, but azimuth is measured from
 * local x axis not y. The range for our phi is
 *  -(full angle)/2 to +(full angle)/2. <BR>
 *
 */

class RadialStripTopology : public CSCStripTopology {
 public:

  /** Constructor from:<BR>
   *    number of strips<BR>
   *    angular width of a strip<BR>
   *    detector height (2 x apothem - we love that word)<BR>
   *    radial distance from symmetry centre of detector to
   *    intersection point of the projected
   *    non-parallel trapezoidal edges.<BR>
   */
  RadialStripTopology( int ns, float aw, float dh, float r );

  /** Destructor
   */
  virtual ~RadialStripTopology() = 0;

  // =========================================================
  // StripTopology interface - implement pure virtual methods
  // =========================================================

  /** LocalPoint on x axis for given 'strip'
   * 'strip' is a float in units of the strip (angular) width
   */
  virtual LocalPoint localPosition(float strip) const;

  /** LocalPoint for a given MeasurementPoint <BR>
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
  virtual LocalPoint localPosition(const MeasurementPoint&) const;

  /** LocalError for a pure strip measurement, where 'strip'
   * is the (float) position (a 'phi' angle wrt y axis) and
   * stripErr2 is the sigma-squared. Both quantities are expressed in
   * units of theAngularWidth of a strip.
   */
  virtual LocalError localError(float strip, float stripErr2) const;

  /** LocalError for a given MeasurementPoint with known MeasurementError.
   * This may be used in Kalman filtering and hence must allow possible
   * correlations between the components.
   */
  virtual LocalError localError(const MeasurementPoint&, const MeasurementError&) const;

  /** Strip in which a given LocalPoint lies. This is a float which
   * represents the fractional strip position within the detector.<BR>
   * Returns zero if the LocalPoint falls at the extreme low edge of the
   * detector or BELOW, and float(nstrips) if it falls at the extreme high
   * edge or ABOVE.
   */
  virtual float strip(const LocalPoint&) const;


  /** Pitch at centre of detector, along local x axis.<BR>
   * BEWARE: Approximation since strip width along x is not
   * constant for a RadialStripTopology. You may really need some
   * other quantity... 
   * This returns arc length for a strip centred on y axis.
   */
  virtual float pitch() const;

  /** Pitch at a given LocalPoint. <BR>
   * BEWARE: like pitch(), not very useful in a
   * RadialStripTopology.
   */
  virtual float localPitch(const LocalPoint&) const;

  /** Angle between strip and symmetry axis (=local y axis)
   * for given strip. <BR>
   * This is like a phi angle but measured clockwise from y axis 
   * rather than counter clockwise from x axis.
   * Note that 'strip' is a float with a continuous range from 0 to 
   * float(nstrips) to cover the whole detector, and the centres of
   * strips correspond to half-integer values 0.5, 1.5, ..., nstrips-0.5
   * whereas values 1, 2, ... nstrips correspond to the upper phi edges of
   * the strips.
   */
  virtual float stripAngle(float strip) const;

  /** Total number of strips covering face of detector
   */
  virtual int nstrips() const; 

  /** Returns height of detector (=extent in local y),
   * just as for a TrapezoidalStripTopology. <BR>
   * BEWARE! The method name seems misleading.<BR>
   */
  virtual float stripLength() const {return theDetHeight;}

  /** Length of a strip passing through a given LocalPpoint
   */
  virtual float localStripLength(const LocalPoint& ) const;


  // =========================================================
  // Topology interface (not already implemented for 
  // StripTopology interface)
  // =========================================================

  virtual MeasurementPoint measurementPosition( const LocalPoint& ) const;

  virtual MeasurementError measurementError( const LocalPoint&, const LocalError& ) const;

  /** Channel number corresponding to a given LocalPoint.<BR>
   * This is effectively an integer version of strip(), with range 0 to
   * nstrips-1.  <BR>
   * LocalPoints outside the detector will be considered
   * as contributing to the edge channels 0 or nstrips-1.
   */
  virtual int channel(const LocalPoint&) const;


  // =========================================================
  // CSCStripTopology interface
  // =========================================================

  /** 
   * Angular width of a each strip in this StripTopology.
   *
   * Note that this isn't in TrapezoidalStripTopology but in
   * OffsetTST, whereas we have it here in the RST and not in
   * the OffsetRST. Confused? Me too. That's what happens when
   * you can't modify a base class.
   */
  virtual float phiPitch(void) const { return theAngularWidth; }

  /** Distance from the intersection of the projections of
   * the extreme edges of the two extreme strips to the symmetry
   * centre of the detector. 
   *
   * WARNINGS!
   * 1) This intersection point does not necessarily fall on
   * the beam axis.
   * 2) For Radial Strips the edges of the extreme strips are NOT
   * collinear with the trapezoidal edges of the chamber volume!
   *
   */
  float centreToIntersection() const { return theCentreToIntersection; }

  /**
   * Local x where centre of strip intersects input local y
   */
  float xOfStrip(int strip, float y) const;

  // =========================================================
  // The RadialStripTopology interface itself
  // =========================================================
  
  friend std::ostream & operator<<(std::ostream &, const RadialStripTopology &);
   /**
   * Effective virtual operator<<
   */
  virtual std::ostream& put(std::ostream& s ) const { return s << *this;}

 protected: 

  /**
   * The nominal RadialStripTopology is perfectly symmetric in azimuth
   * about the local y axis. This rotates that symmetric configuration
   * by some fraction of the angular strip width.<BR>
   * The neighbouring layers in a 6-layer Endcap Muon CSC are alternately
   * rotated by half a strip width w.r.t. each other. 
   * This method can implement that offset.
   */
  //  virtual float shiftOffset( float fractionOfStrip );

  /**
   * Enable derived classes to do some work
   */
  float angularWidth() const { return theAngularWidth;}
  float detHeight() const { return theDetHeight;}
  float phiOfOneEdge() const { return thePhiOfOneEdge;}

 private:
  // The RST has theNumberOfStrips since the TST does
  // In an ideal world it would be in a common base class but
  // I don't have the freedom to make changes to the TST
  int   theNumberOfStrips; // total no. of strips in detector
  float theAngularWidth;   // angle subtended by each strip = phi pitch

  // theDetHeight caches twice the apothem for the trapezoidal detector
  // This could come from TrapezoidalPlaneBounds base class of MELG
  // via TPB::length() and it is also cached in CSCLayerGeometry.

  float theDetHeight; 

  // theCentreToIntersection caches distance from centre of detector face
  // to intersection of non-parallel edges. This is also whereStripsMeet
  // in the CSCLayerGeometry.

  float theCentreToIntersection;

  float thePhiOfOneEdge;  // local 'phi' of one edge of detector (negative!)

};


#endif


