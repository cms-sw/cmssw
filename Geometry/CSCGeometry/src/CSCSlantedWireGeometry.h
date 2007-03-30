#ifndef CSC_SLANTED_WIRE_GEOMETRY_H
#define CSC_SLANTED_WIRE_GEOMETRY_H

/** \class CSCSlantedWireGeometry
 * A concrete CSCWireGeometry in which wires are slanted,
 * i.e. they have a fixed, non-zero angle w.r.t. local x axis.
 *
 * \author Tim Cox
 *
 */

#include "Geometry/CSCGeometry/interface/CSCWireGeometry.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class CSCSlantedWireGeometry : public CSCWireGeometry {
 public:
  virtual ~CSCSlantedWireGeometry() {}

  /**
   * Constructor from wire spacing and wire angle
   */
  CSCSlantedWireGeometry( double wireSpacing, double yOfFirstWire, 
         double narrow, double wide, double length,
         float wireAngle );
 
  /**
   * The angle of the wires w.r.t local x axis (in radians)
   */
  float wireAngle() const { return theWireAngle; }

  /**
   * The nearest (virtual) wire to a given LocalPoint.
   * Beware that this wire might not exist or be read out!
   */
  int nearestWire(const LocalPoint& lp) const;

  /**
   * Local y of a given wire 'number' (float) at given x
   */
  float yOfWire(float wire, float x=0.) const;

  /**
   * Clone to handle correct copy of component objects referenced
   * by base class pointer.
   */
  CSCWireGeometry* clone() const {
    return new CSCSlantedWireGeometry(*this);
  }

 private:
  float theWireAngle;
  float cosWireAngle;
  float sinWireAngle;
  float theWireOffset; // local y of first wire * cos(wire angle)

};

#endif
