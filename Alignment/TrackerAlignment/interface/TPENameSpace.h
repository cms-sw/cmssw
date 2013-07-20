#ifndef Alignment_CommonAlignment_TPENameSpace_H
#define Alignment_CommonAlignment_TPENameSpace_H

/** \namespace tpe
 *
 *  Namespace for numbering components in Forward Pixel.
 *
 *  A system to number a component within its parent; starts from 1.
 *
 *  $Date: 2013/01/07 19:44:30 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "CondFormats/Alignment/interface/Definitions.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

namespace align
{
  namespace tpe
  {
    const unsigned int bpqd = 6; // no. of blades per quarter disk

    /// Module number increases with rho; from 1 to 4.
    inline unsigned int moduleNumber(align::ID, const TrackerTopology*);

    /// Panel number is 1 for 4 modules, 2 for 3 modules.
    inline unsigned int panelNumber(align::ID, const TrackerTopology*);

    /// Blade number increases from 1 at the top to 12 at the bottom
    /// of each half disk.
    inline unsigned int bladeNumber(align::ID, const TrackerTopology*);

    /// Half disk number increases with |z| from 1 to 3.
    inline unsigned int halfDiskNumber(align::ID, const TrackerTopology*);

    /// Half cylinder number is 1 at left side (-x) and 2 at right side (+x).
    inline unsigned int halfCylinderNumber(align::ID, const TrackerTopology*);

    /// Endcap number is 1 for -z and 2 for +z.
    inline unsigned int endcapNumber(align::ID, const TrackerTopology*);
  }
}

unsigned int align::tpe::moduleNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->pxfModule(id);
}

unsigned int align::tpe::panelNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->pxfPanel(id);
}

unsigned int align::tpe::bladeNumber(align::ID id, const TrackerTopology* tTopo)
{
  unsigned int b = tTopo->pxfBlade(id); // 1 to 24 in increasing phi

// Blade in 1st quadrant: number = bpqd + 1 - b     (1 to bpqd)
// Blade in 2nd quadrant: number = b - bpqd         (1 to bpqd)
// Blade in 3rd quadrant: number = b - bpqd         (bpqd + 1 to 2 * bpqd)
// Blade in 4th quadrant: number = 5 * bpqd + 1 - b (bpqd + 1 to 2 * bpqd)

  return b > 3 * bpqd ? // blade in 4th quadrant
    5 * bpqd + 1 - b :
    (b > bpqd ? // blade not in 1st quadrant
     b - bpqd : bpqd + 1 - b);
}

unsigned int align::tpe::halfDiskNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->pxfDisk(id);
}

unsigned int align::tpe::halfCylinderNumber(align::ID id, const TrackerTopology* tTopo)
{
  unsigned int b = tTopo->pxfBlade(id); // 1 to 24 in increasing phi

  return b > bpqd && b <= 3 * bpqd ? 1 : 2;
}

unsigned int align::tpe::endcapNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->pxfSide(id);
}

#endif
