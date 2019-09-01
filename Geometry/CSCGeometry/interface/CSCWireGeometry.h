#ifndef CSC_WIRE_GEOMETRY_H
#define CSC_WIRE_GEOMETRY_H

/** \class CSCWireGeometry
 * An ABC defining interface for geometry related to angle
 * which wires of a detector modelled by a WireTopology
 * make with the local x axis.
 *
 * \author Tim Cox
 *
 */

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include <vector>
#include <utility>  // for std::pair

class CSCWireGeometry {
public:
  virtual ~CSCWireGeometry() {}

  /**
   * Constructor from wire spacing (in cm)
   */
  CSCWireGeometry(
      double wireSpacing, double yOfFirstWire, double narrowWidthOfPlane, double wideWidthOfPlane, double lengthOfPlane)
      : theWireSpacing(wireSpacing),
        theYOfFirstWire(yOfFirstWire),
        theNarrowWidthOfPlane(narrowWidthOfPlane),
        theWideWidthOfPlane(wideWidthOfPlane),
        theLengthOfPlane(lengthOfPlane) {}

  /**
   * The spacing between wires (cm)
   */
  double wireSpacing() const { return theWireSpacing; }

  /** 
   * The local y of the first wire
   */
  double yOfFirstWire() const { return theYOfFirstWire; }

  /** 
   * Extent of wire plane at narrow end of trapezoid
   */
  double narrowWidthOfPlane() const { return theNarrowWidthOfPlane; }

  /** 
   * Extent of wire plane at wide end of trapezoid
   */
  double wideWidthOfPlane() const { return theWideWidthOfPlane; }

  /** 
   * Extent of wire plane along long axis of trapezoid
   */
  double lengthOfPlane() const { return theLengthOfPlane; }

  /**
   * The angle of the wires w.r.t local x axis (in radians)
   */
  virtual float wireAngle() const = 0;

  /**
   * The nearest (virtual) wire to a given LocalPoint.
   * Beware that this wire might not exist or be read out!
   */
  virtual int nearestWire(const LocalPoint& lp) const = 0;

  /**
   * Local y of a given wire 'number' (float) at given x
   */
  virtual float yOfWire(float wire, float x = 0.) const = 0;

  /**
   * Allow proper copying of derived classes via base pointer
   */
  virtual CSCWireGeometry* clone() const = 0;

  /** 2D point of intersection of two straight lines defined by <BR>
   *  y = m1*x + c1 and y = m2*x + c2 <BR>
   *  (in local coordinates x, y)
   */
  LocalPoint intersection(float m1, float c1, float m2, float c2) const;

  /** Return 2-dim local coords of the two ends of a wire
   *
   *  The returned value is a pair of LocalPoints. 
   */
  std::pair<LocalPoint, LocalPoint> wireEnds(float wire) const;

  /** Return mid-point of a wire in local coordinates, and its length
   *  across the chamber volume, in a vector as x, y, length
   */
  std::vector<float> wireValues(float wire) const;

  /**
   * Return slope and intercept of straight line representing a wire in 2-dim local coordinates.
   *
   * The return value is a pair p with p.first = m, p.second = c, where y=mx+c.
   */
  std::pair<float, float> equationOfWire(float wire) const;

  /**
   * Return pair containing y extremes of wire-plane: p.first = low y, p.second= high y
   *
   * This is supposed to approximate the 'sensitive' region covered by wires (and strips) 
   * but there is no sophisticated handling of edge effects, or attempt to estimate a
   * precise region overlapped by both wires and strips.
   */
  std::pair<float, float> yLimitsOfWirePlane() const;

private:
  double theWireSpacing;
  double theYOfFirstWire;  // local y
  double theNarrowWidthOfPlane;
  double theWideWidthOfPlane;
  double theLengthOfPlane;
};

#endif
