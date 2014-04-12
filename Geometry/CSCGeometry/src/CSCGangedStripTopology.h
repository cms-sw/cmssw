#ifndef ORED_OFFSET_RST_H
#define ORED_OFFSET_RST_H

/** \class CSCGangedStripTopology
 *  A concrete CSCStripTopology in which strips are ganged, as in ME1A chambers.
 *
 *  \author Tim Cox
 *
 */

#include "Geometry/CSCGeometry/interface/CSCStripTopology.h"

class CSCGangedStripTopology : public CSCStripTopology
{
public:

  CSCGangedStripTopology(const CSCStripTopology & topology, int numberOfGangedStrips ) 
    : CSCStripTopology(topology), theNumberOfGangedStrips(numberOfGangedStrips) {}

  ~CSCGangedStripTopology() {}

  /** 
   * Return channel corresponding to a LocalPoint.
   * (Count from 1)
   */
  int channel(const LocalPoint& lp) const {
    return (int) (CSCRadialStripTopology::strip(lp)) % theNumberOfGangedStrips + 1;
  }

  /** 
   * Return channel corresponding to a strip.
   * (Count from 1).
   */
  int channel(int strip) const {
    while(strip > theNumberOfGangedStrips) strip -= theNumberOfGangedStrips;
    while(strip <= 0) strip += theNumberOfGangedStrips;
    return strip;
  }

  /**
   * Clone to handle correct copy of component objects referenced
   * by base class pointer.
   * If gcc could handle it, should be
   *   virtual CSCGangedStripTopology* clone() const
   */
  CSCStripTopology* clone() const {
    return new CSCGangedStripTopology(*this);
  }

  /**
   * Implement CSCStripTopology interface for its op<<
   */
  std::ostream& put ( std::ostream& os ) const {
    return os << "CSCGangedStripTopology";
  }

 private:
  int theNumberOfGangedStrips;
};

#endif


