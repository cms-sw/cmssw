#ifndef CSC_STRIP_TOPOLOGY_H
#define CSC_STRIP_TOPOLOGY_H

#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <iosfwd>

/** \class CSCStripTopology
 * ABC interface for all endcap muon CSC strip topologies.
 *
 * \author Tim Cox
 *
 */

class CSCStripTopology : public virtual StripTopology {
public:

  virtual ~CSCStripTopology() {}

  virtual float stripOffset() const = 0;
  virtual float phiPitch() const = 0;
  virtual CSCStripTopology* clone() const = 0;
  virtual int channel(int) const = 0;
  virtual int channel(const LocalPoint&) const = 0; // also in  Topology base
  virtual float centreToIntersection() const = 0;
  virtual int nearestStrip(const LocalPoint&) const;
  virtual float strip(const LocalPoint&) const = 0;
  virtual float xOfStrip(int strip, float y) const = 0;
  virtual std::ostream& put(std::ostream&) const = 0;
  friend std::ostream& operator<<(std::ostream& s, const CSCStripTopology& r);
};

#endif
