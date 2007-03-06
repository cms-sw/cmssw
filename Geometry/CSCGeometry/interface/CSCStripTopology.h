#ifndef CSC_STRIP_TOPOLOGY_H
#define CSC_STRIP_TOPOLOGY_H

#include "Geometry/CSCGeometry/interface/OffsetRadialStripTopology.h"
#include <iosfwd>

/** \class CSCStripTopology
 * ABC interface for all endcap muon CSC radial strip topologies. <BR>
 * In the Endcap Muon CSCs, the cathode strips are not strictly fan-shaped, 
 * each subtending a constant azimuthal angle, and project to a point. 
 * In every station and ring except for ME13 the nominal (perfect) geometry 
 * has this point of intersection (approximately) on the beam line. 
 * That constraint is unused as far as possible in order
 * to allow non-perfect geometry and misalignment scenarios.<BR>
 * Note that the base class RST is concrete but this class is again abstract
 * (for both historical and operational reasons.) <BR>
 * Alternate strip layers in each CSC are relatively offset by half-a-strip width
 * so the CSCStripTopology must be an OffsetRadialStripTopology, rather than
 * a simple RadialStripTopology in which the long symmetry axis of the plane
 * of strips is aligned with the local y axis of the detector. <BR>
 *
 * \author Tim Cox
 *
 */

class CSCStripTopology : public  OffsetRadialStripTopology {
public:

  /** 
   * Constructor from:<BR>
   *    number of strips<BR>
   *    angular width of a strip<BR>
   *    detector height (2 x apothem - we love that word)<BR>
   *    radial distance from symmetry centre of detector to the point at which 
   *    the outer edges of the two extreme strips (projected) intersect.<BR>
   *    offset of y symmetry axis from local y as fraction of angular strip width. <BR>
   */
  CSCStripTopology( int ns, float aw, float dh, float r, float aoff );

  virtual ~CSCStripTopology();

  virtual CSCStripTopology* clone() const = 0;

  /**
   * Virtual output function which is used to implement op<<
   */
  virtual std::ostream& put(std::ostream&) const = 0;

  friend std::ostream& operator<<(std::ostream& s, const CSCStripTopology& r);
};

#endif
