#ifndef CSC_WIRE_TOPOLOGY_H
#define CSC_WIRE_TOPOLOGY_H

/** \class CSCWireTopology
 * A concrete class derived from WireTopology, to handle
 * wire (group) geometry functionality for endcap muon CSCs.
 *
 * \author Tim Cox
 *
 */

#include "Geometry/CSCGeometry/interface/WireTopology.h"
#include "Geometry/CSCGeometry/interface/CSCWireGeometry.h"
#include "Geometry/CSCGeometry/src/CSCWireGrouping.h"
#include "Geometry/CSCGeometry/interface/CSCWireGroupPackage.h"
#include <utility>  // for std::pair

class CSCWireTopology : public WireTopology {
public:
  ~CSCWireTopology() override;

  /**
   * Constructor from endcap muon CSC wire geometry specs
   */

  CSCWireTopology(const CSCWireGroupPackage& wg, double yOfFirstWire, float wireAngleInDegrees);
  /**
   * Copy constructor
   */
  CSCWireTopology(const CSCWireTopology&);

  /**
   * Assignment op
   */
  CSCWireTopology& operator=(const CSCWireTopology&);

  /** 
   * Topology interface, but not implemented for CSCWireTopology (yet!)
   */

  LocalPoint localPosition(const MeasurementPoint&) const override;
  LocalError localError(const MeasurementPoint&, const MeasurementError&) const override;
  MeasurementPoint measurementPosition(const LocalPoint&) const override;
  MeasurementError measurementError(const LocalPoint&, const LocalError&) const override;

  /**
   * 'channel' is wire group number from 1 to no. of groups.
   * Return 0 if out-of-range or in a dead region
   */
  int channel(const LocalPoint& p) const override;

  /**
   * WireTopology interface
   */

  /** 
   * The wire spacing (in cm)
   */
  double wireSpacing() const { return theWireGeometry->wireSpacing(); }

  /**
   * The wire pitch. This is the wire spacing but
   * old-timers like the word 'pitch'.
   */
  float wirePitch() const override { return static_cast<float>(wireSpacing()); }

  /**
   * The angle of the wires w.r.t local x axis (in radians)
   */
  float wireAngle() const override { return theWireGeometry->wireAngle(); }

  /**
   * The nearest (virtual) wire to a given LocalPoint.
   * Beware that this wire might not exist or be read out!
   */
  int nearestWire(const LocalPoint& lp) const override { return theWireGeometry->nearestWire(lp); }

  /**
   * Local y of a given wire 'number' (float) at given x
   */
  float yOfWire(float wire, float x = 0.) const { return theWireGeometry->yOfWire(wire, x); }

  /**
   * Width of wire plane at narrow end of trapezoid
   */
  double narrowWidthOfPlane() const { return theWireGeometry->narrowWidthOfPlane(); }

  /**
   * Width of wire plane at wide end of trapezoid
   */
  double wideWidthOfPlane() const { return theWireGeometry->wideWidthOfPlane(); }

  /**
   * Length/height of wire plane along long axis of trapezoid (local y direction)
   */
  double lengthOfPlane() const { return theWireGeometry->lengthOfPlane(); }

  /**
   * Wire group interface
   */

  /**
   * Total number of (virtual) wires.
   * Some wires may not be implemented in the hardware.
   * This is the number which would fill the region covered
   * by wires, assuming the constant wire spacing.
   */
  int numberOfWires() const override { return theWireGrouping->numberOfWires(); }

  /**
   * How many wire groups
   */
  int numberOfWireGroups() const { return theWireGrouping->numberOfWireGroups(); }

  /**
   * How many wires in a wiregroup
   */
  int numberOfWiresPerGroup(int wireGroup) const { return theWireGrouping->numberOfWiresPerGroup(wireGroup); }

  /**
   * Wire group containing a given wire
   */
  int wireGroup(int wire) const { return theWireGrouping->wireGroup(wire); }

  /**
   * Middle of wire-group.
   * This is the central wire no. for a group with an odd no. of wires.
   * This is a pseudo-wire no. for a group with an even no. of wires.
   * Accordingly, it is non-integer.
   */
  float middleWireOfGroup(int wireGroup) const { return theWireGrouping->middleWireOfGroup(wireGroup); }

  /**
   * Extended interface which 'mixes' WireGrouping and WireGeometry info
   */

  /**
   * Local y of a given wire group at given x
   */
  float yOfWireGroup(int wireGroup, float x = 0.) const;

  /**
   * The measurement resolution from wire groups (in cm.)
   * This approximates the measurement resolution in the local
   * y direction but may be too small by a factor of up to 1.26
   * due to stripAngle contributions which are neglected here.
   */
  float yResolution(int wireGroup = 1) const;

  /** 
   * Extent of wire plane (width normal to wire direction). <BR>
   * Note that for ME11 this distance is not along local y! <BR>
   * cf. lengthOfPlane() which should be the same for all chambers but ME11.
   */
  double extentOfWirePlane() const { return wireSpacing() * (numberOfWires() - 1); }

  /**
   * Return local (x,y) coordinates of the two ends of a wire
   * across the extent of the wire plane.
   * The returned value is a pair of LocalPoints.
   */
  std::pair<LocalPoint, LocalPoint> wireEnds(float wire) const { return theWireGeometry->wireEnds(wire); }

  /** Return mid-point of a wire in local coordinates, and its length
   *  across the chamber volume, in a vector as x, y, length
   */
  std::vector<float> wireValues(float wire) const { return theWireGeometry->wireValues(wire); }

  /**
   * Return slope and intercept of straight line representing a wire in 2-dim local coordinates.
   *
   * The return value is a pair p with p.first = m, p.second = c, where y=mx+c.
   */
  std::pair<float, float> equationOfWire(float wire) const;

  /**
   * Reset input y to lie within bounds of wire plane at top and bottom.
   */
  float restrictToYOfWirePlane(float y) const;

  /** 
   * Returns true if arg falls within y limits of wire plane; false otherwise.
   */
  bool insideYOfWirePlane(float y) const;

private:
  CSCWireGrouping* theWireGrouping;  // handles grouping of wires for read out
  CSCWireGeometry* theWireGeometry;  // handles non-zero angle w.r.t x axis

  double theAlignmentPinToFirstWire;  //@@ Not sure this is actually required!
};

#endif
