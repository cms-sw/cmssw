#ifndef ORED_OFFSET_RST_H
#define ORED_OFFSET_RST_H

/** \class ORedOffsetRST
 *  In ME1a strips are ganged together. 
 *  This is an OffsetRadialStripTopology to handle this.
 *
 *  \author Tim Cox
 *
 */

#include "Geometry/CSCGeometry/src/AbsOffsetRadialStripTopology.h"

class ORedOffsetRST : public AbsOffsetRadialStripTopology
{
public:

  ORedOffsetRST(const AbsOffsetRadialStripTopology & topology, 
                      int staggering ) 
    : AbsOffsetRadialStripTopology(topology),
    theStaggering(staggering) {}

  ~ORedOffsetRST() {}

  /** 
   * Return channel corresponding to a LocalPoint.
   * (Count from 1)
   */
  int channel(const LocalPoint& lp) const {
    return (int) (RadialStripTopology::strip(lp)) % theStaggering + 1;
  }

  /** 
   * Return channel corresponding to a strip.
   * (Count from 1).
   */
  int channel(int strip) const {
    while(strip > theStaggering) strip -= theStaggering;
    while(strip <= 0) strip += theStaggering;
    return strip;
  }

  /**
   * Clone to handle correct copy of component objects referenced
   * by base class pointer.
   * If gcc could handle it, should be
   *   virtual ORedOffsetRST* clone() const
   */
  CSCStripTopology* clone() const {
    return new ORedOffsetRST(*this);
  }

 private:
  int theStaggering;
};

#endif


