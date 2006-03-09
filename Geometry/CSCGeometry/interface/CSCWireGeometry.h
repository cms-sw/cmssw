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

//@@ Forward declaration alone didn't work in ORCA.. But now?
//class LocalPoint;
#include "Geometry/Vector/interface/LocalPoint.h"

class CSCWireGeometry {
 public:
  virtual ~CSCWireGeometry() {}

  /**
   * Constructor from wire spacing (in cm)
   */
  CSCWireGeometry( double wireSpacing, double yOfFirstWire ) :
     theWireSpacing( wireSpacing ), theYOfFirstWire( yOfFirstWire ) {}

  /**
   * The spacing between wires (cm)
   */
  double wireSpacing() const {
    return theWireSpacing; }

  /** 
   * The local y of the first wire
   */
  double yOfFirstWire() const {
    return theYOfFirstWire; }

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
  virtual float yOfWire(float wire, float x=0.) const = 0;

  /**
   * Allow proper copying of derived classes via base pointer
   */
  virtual CSCWireGeometry* clone() const = 0;

 private:
  double theWireSpacing;
  double theYOfFirstWire; // local y
};

#endif
