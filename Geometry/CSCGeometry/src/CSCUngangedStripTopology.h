#ifndef OFFSET_RADIAL_STRIP_TOPOLOGY_H
#define OFFSET_RADIAL_STRIP_TOPOLOGY_H

/** \class CSCUngangedStripTopology
 *  A concrete CSCStripTopology with unganged strips (normal CSC case.)
 *
 *  \author Tim Cox
 * 
 */

#include "Geometry/CSCGeometry/interface/CSCStripTopology.h"

class CSCUngangedStripTopology : public CSCStripTopology
{
public:

  CSCUngangedStripTopology( int numberOfStrips, float stripPhiPitch,
       float detectorHeight, float whereStripsMeet, float stripOffset, float yCentre):
    CSCStripTopology( numberOfStrips, stripPhiPitch, 
       detectorHeight, whereStripsMeet, stripOffset, yCentre ){}

  ~CSCUngangedStripTopology(){}

  /** 
   * Return channel corresponding to a LocalPoint.
   * (but we count from 1 whereas RST counts from 0.)
   */
  int channel(const LocalPoint& lp) const {
    return CSCRadialStripTopology::channel(lp) + 1;
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
   *   virtual CSCUngangedStripTopology* clone() const
   */
  CSCStripTopology* clone() const {
    return new CSCUngangedStripTopology(*this);
  }

  /**
   * Implement CSCStripTopology interface for its op<<
   */
  std::ostream& put ( std::ostream& os ) const {
    return os << "CSCUngangedStripTopology";
  }
};

#endif

