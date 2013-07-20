#ifndef Alignment_CommonAlignment_TOBNameSpace_H
#define Alignment_CommonAlignment_TOBNameSpace_H

/** \namespace tob
 *
 *  Namespace for numbering components in Tracker Outer Barrel.
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
  namespace tob
  {
     /// Module number increases with |z| from 1 to 6.
    inline unsigned int moduleNumber(align::ID, const TrackerTopology*);

    /// Rod number increases with phi.
    inline unsigned int rodNumber(align::ID, const TrackerTopology*);

    /// Layer number increases with rho from 1 to 6.
    inline unsigned int layerNumber(align::ID, const TrackerTopology*);

    /// HalfBarrel number is 1 at -z side and 2 at +z side.
    inline unsigned int halfBarrelNumber(align::ID, const TrackerTopology*);
  }
}

unsigned int align::tob::moduleNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tobModule(id);
}

unsigned int align::tob::rodNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tobRodInfo(id)[1];
}

unsigned int align::tob::layerNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tobLayer(id);
}

unsigned int align::tob::halfBarrelNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tobRodInfo(id)[0];
}

#endif
