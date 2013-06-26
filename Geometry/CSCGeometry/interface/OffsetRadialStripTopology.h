#ifndef ABS_OFFSET_RADIAL_STRIP_TOPOLOGY_H
#define ABS_OFFSET_RADIAL_STRIP_TOPOLOGY_H

/** \class OffsetRadialStripTopology
 *  ABC defining  RadialStripTopology with shifted offset so that it
 *  is not centred on local y (of parent chamber)
 *
 *  The offset is specified as a fraction of the strip angular width.
 *
 *  \author Tim Cox
 * 
 */

#include "Geometry/CommonTopologies/interface/CSCRadialStripTopology.h"
#include <iosfwd>

class OffsetRadialStripTopology : public CSCRadialStripTopology
{
public:

  /** Constructor
   *  Note that yCentre is local y of symmetry centre of strip plane
   *  _before_ the rotation shift: it is passed directly to RST base.
   */
  OffsetRadialStripTopology( int numberOfStrips, float stripPhiPitch,
     float detectorHeight, float radialDistance, float stripOffset, float yCentre);

  virtual ~OffsetRadialStripTopology(){};

  /** Fraction of a strip offset of layer relative to
   *  symmetry axis (local y). (This is an _angular_ value)
   */
  virtual float stripOffset( void ) const { return theStripOffset; }

  /** LocalPoint for a given strip
   */
  virtual LocalPoint localPosition(float strip) const {
    // Pass through to base class since otherwise it is shadowed by the localPosition(const MP&).
    // Note that base class version is OK because it uses stripAngle() which is overridden in ORST!
    // Also note that xOfStrip from base class RST also works for ORST for the same reason.
    return CSCRadialStripTopology::localPosition( strip );
  }

  /** LocalPoint for a given MeasurementPoint <BR>
   *
   * What's a MeasurementPoint?  <BR>
   * A MeasurementPoint is a 2-dim object, with the 1st dim specifying the angular position
   * in strip widths, and the 2nd dim specifying the fractional distance alone a strip.<BR>
   *
   * Thus the 1st dimension measures the
   * angular position wrt central line of symmetry of detector,
   * in units of strip (angular) widths (range 0 to total angle subtended
   * by a detector).
   * The 2nd dimension measures
   * the fractional position along the strip (range -0.5 to +0.5).<BR>
   *
   * BEWARE! The components are not Cartesian.<BR>
   * BEWARE! Neither coordinate may correspond to either local x or local y.<BR>
   * BEWARE! This involves ONLY strip-related measurements, not CSC wires!
   */
  virtual LocalPoint localPosition(const MeasurementPoint&) const;

  /**
   * MeasurementPoint corresponding to given LocalPoint
   */
  virtual MeasurementPoint measurementPosition( const LocalPoint& ) const;

  /** Strip in which a given LocalPoint lies. This is a float which
   * represents the fractional strip position within the detector.<BR>
   * Returns zero if the LocalPoint falls at the extreme low edge of the
   * detector or BELOW, and float(nstrips) if it falls at the extreme high
   * edge or ABOVE.
   */
  virtual float strip(const LocalPoint&) const;

  /**
   * Angle between strip and local y axis (measured clockwise from y axis)
   */
  float stripAngle(float strip) const;

  /**
   * Channel number corresponding to a strip or a LocalPoint.
   * Sometimes more than one strip is OR'ed into one channel.
   */
  virtual int channel(int strip) const = 0;
  virtual int channel(const LocalPoint& lp) const = 0;

  friend std::ostream & operator<<(std::ostream &, const OffsetRadialStripTopology &);

 private:
  /**
   * Transform from coordinates wrt strip plane symmetry axes to
   * local coordinates
   */
  LocalPoint toLocal(float xprime, float yprime) const;
  /**
   * Transform from local coordinates to coordinates wrt strip plane
   * symmetry axes
   */
  LocalPoint toPrime(const LocalPoint&) const;

  float theStripOffset; // fraction of a strip offset from sym about y
  float theCosOff; // cosine of angular offset
  float theSinOff; // sine of angular offset
};

#endif

