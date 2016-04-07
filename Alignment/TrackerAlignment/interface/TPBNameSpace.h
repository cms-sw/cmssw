#ifndef Alignment_TrackerAlignment_TPBNameSpace_H
#define Alignment_TrackerAlignment_TPBNameSpace_H

/** \namespace tpb
 *
 *  Namespace for numbering components in Barrel Pixel.
 *
 *  A system to number a component within its parent; starts from 1.
 *
 *  $Date: 2007/10/18 09:57:10 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 *
 *  Last Update: Max Stark
 *         Date: Fri, 05 Feb 2016 17:21:33 CET
 */

#include "CondFormats/Alignment/interface/Definitions.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

namespace align
{
  namespace tpb
  {
    /// Number of ladders for each quarter cylinder.
    std::vector<unsigned int> lpqc;

    /// Module number increases with z from 1 to 8.
    inline unsigned int moduleNumber(align::ID, const TrackerTopology*);

    /// Ladder number increases from 1 at the top to 2 * lpqc at the bottom
    /// of each half cylinder.
    inline unsigned int ladderNumber(align::ID, const TrackerTopology*);

    /// Layer number increases with rho from 1 to 3.
    inline unsigned int layerNumber(align::ID, const TrackerTopology*);

    /// Half barrel number is 1 at left side (-x) and 2 at right side (+x).
    inline unsigned int halfBarrelNumber(align::ID, const TrackerTopology*);

    /// Barrel number is 1 for all align::ID's which belong to this barrel
    inline unsigned int barrelNumber(align::ID, const TrackerTopology*);
  }
}

unsigned int align::tpb::moduleNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->pxbModule(id);
}

unsigned int align::tpb::ladderNumber(align::ID id, const TrackerTopology* tTopo)
{
  unsigned int l = tTopo->pxbLadder(id); // increases with phi
  unsigned int c = tTopo->pxbLayer(id) - 1;

// Ladder in 1st quadrant: number = lpqc + 1 - l     (1 to lpqc)
// Ladder in 2nd quadrant: number = l - lpqc         (1 to lpqc)
// Ladder in 3rd quadrant: number = l - lpqc         (lpqc + 1 to 2 * lpqc)
// Ladder in 4th quadrant: number = 5 * lpqc + 1 - l (lpqc + 1 to 2 * lpqc)

  return l > 3 * lpqc[c] ? 5 * lpqc[c] + 1 - l :  // ladder in 4th quadrant
           (l > lpqc[c] ? l - lpqc[c] :           // ladder not in 1st quadrant
             lpqc[c] + 1 - l);
}

unsigned int align::tpb::layerNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->pxbLayer(id);
}

unsigned int align::tpb::halfBarrelNumber(align::ID id, const TrackerTopology* tTopo)
{
  unsigned int l = tTopo->pxbLadder(id); // increases with phi
  unsigned int c = tTopo->pxbLayer(id) - 1;

  return l > lpqc[c] && l <= 3 * lpqc[c] ? 1 : 2;
}

unsigned int align::tpb::barrelNumber(align::ID, const TrackerTopology*)
{
  return 1;
}

#endif
