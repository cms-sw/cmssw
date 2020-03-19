#ifndef WIRE_TOPOLOGY_H
#define WIRE_TOPOLOGY_H

/** \class WireTopology
 * An ABC for detectors using wires rather than strips.
 *
 * Extend the Topology interface
 * to supply required wire geometry functionality.
 *
 * \author Tim Cox
 *
 */

#include "Geometry/CommonTopologies/interface/Topology.h"

class WireTopology : public Topology {
public:
  ~WireTopology() override {}

  /**
   * How many wires
   */
  virtual int numberOfWires() const = 0;

  /**
   * The angle (in radians) of (any) wire wrt local x-axis.
   */
  virtual float wireAngle() const = 0;

  /**
   * The distance (in cm) between wires
   */
  virtual float wirePitch() const = 0;

  /**
   * Wire nearest a given local point
   */
  virtual int nearestWire(const LocalPoint &) const = 0;

private:
};

#endif
