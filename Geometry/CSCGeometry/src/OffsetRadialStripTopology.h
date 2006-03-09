#ifndef OFFSET_RADIAL_STRIP_TOPOLOGY_H
#define OFFSET_RADIAL_STRIP_TOPOLOGY_H

/** \class OffsetRadialStripTopology
 *  A concrete RadialStripTopology with shifted offset so that it
 *  is not centred on local y (of parent chamber.)
 *  The offset is specified as a fraction of the strip angular width.
 *
 *  \author Tim Cox
 * 
 */

#include "Geometry/CSCGeometry/src/AbsOffsetRadialStripTopology.h"
class CSCStripTopology;

class OffsetRadialStripTopology : public AbsOffsetRadialStripTopology
{
public:

  OffsetRadialStripTopology( int numberOfStrips, float stripPhiPitch,
       float detectorHeight, float radialDistance, float stripOffset):
    AbsOffsetRadialStripTopology( numberOfStrips, stripPhiPitch, 
       detectorHeight, radialDistance, stripOffset){}

  ~OffsetRadialStripTopology(){}

  /** 
   * Return channel corresponding to a LocalPoint.
   * (Count from 1)
   */
  int channel(const LocalPoint& lp) const {
    return RadialStripTopology::channel(lp) + 1;
  }

  /** 
   * Return channel corresponding to a strip.
   * (Count from 1).
   */
  int channel(int strip) const {return strip;}

  /**
   * Clone to handle correct copy of component objects referenced
   * by base class pointer.
   * If gcc could handle it, should be
   *   virtual OffsetRadialStripTopology* clone() const
   */
  CSCStripTopology* clone() const {
    return new OffsetRadialStripTopology(*this);
  }

};

#endif

