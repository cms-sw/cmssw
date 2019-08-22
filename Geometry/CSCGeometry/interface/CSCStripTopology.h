#ifndef CSC_STRIP_TOPOLOGY_H
#define CSC_STRIP_TOPOLOGY_H

#include "Geometry/CSCGeometry/interface/OffsetRadialStripTopology.h"
#include <iosfwd>
#include <utility>  // for std::pair

/** \class CSCStripTopology
 * ABC interface for all endcap muon CSC radial strip topologies. <BR>
 * In the Endcap Muon CSCs, the cathode strips are strictly fan-shaped, 
 * each subtending a constant azimuthal angle, and project to a point. 
 * In every station and ring except for ME13 the nominal (perfect) geometry 
 * has this point of intersection (approximately) on the beam line. 
 * That constraint is unused as far as possible in order
 * to allow non-perfect geometry and misalignment scenarios.<BR>
 * Note that the base class RST is concrete but this class is again abstract
 * (for both historical and operational reasons.) <BR>
 * Alternate strip layers in each CSC are relatively offset by half-a-strip width
 * (except in ME11)
 * so the CSCStripTopology must be an OffsetRadialStripTopology, rather than
 * a simple RadialStripTopology in which the long symmetry axis of the plane
 * of strips is aligned with the local y axis of the detector. <BR>
 *
 * \author Tim Cox
 *
 */

class CSCStripTopology : public OffsetRadialStripTopology {
public:
  /** 
   * Constructor
   *    \param ns number of strips
   *    \param aw angular width of a strip
   *    \param dh detector height (extent of strip plane along long symmetry axis))
   *    \param r radial distance from symmetry centre of strip plane to the point at which 
   *    the outer edges of the two extreme strips (projected) intersect.
   *    \param aoff offset of y symmetry axis from local y as fraction of angular strip width.
   *    \param ymid local y of symmetry centre of strip plane _before_ it is offset. <BR>
   */
  CSCStripTopology(int ns, float aw, float dh, float r, float aoff, float ymid);

  ~CSCStripTopology() override;

  /**
   * Return slope and intercept of straight line representing (centre-line of) a strip in 2-dim local coordinates.
   *
   * The return value is a pair p with p.first = m, p.second = c, where y=mx+c.
   */
  std::pair<float, float> equationOfStrip(float strip) const;

  /**
   * Return local y limits of strip plane
   */
  std::pair<float, float> yLimitsOfStripPlane() const;

  virtual CSCStripTopology* clone() const = 0;

  /**
   * Virtual output function which is used to implement op<<
   */
  virtual std::ostream& put(std::ostream&) const = 0;

  friend std::ostream& operator<<(std::ostream& s, const CSCStripTopology& r);
};

#endif
