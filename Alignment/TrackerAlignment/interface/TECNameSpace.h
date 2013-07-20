#ifndef Alignment_CommonAlignment_TECNameSpace_H
#define Alignment_CommonAlignment_TECNameSpace_H

/** \namespace tec
 *
 *  Namespace for numbering components in Tracker Endcaps.
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
  namespace tec
  {
     /// Module number increases (decreases) with phi for +z (-z) endcap.
    inline unsigned int moduleNumber(align::ID, const TrackerTopology*);

    /// Ring number increases with rho.
    inline unsigned int ringNumber(align::ID, const TrackerTopology*);

    /// Petal number increases with phi from 1 to 8.
    inline unsigned int petalNumber(align::ID, const TrackerTopology*);

    /// Side number is 1 for back disk and 2 for front (towards IP).
    inline unsigned int sideNumber(align::ID, const TrackerTopology*);

    /// Disk number increases with |z| from 1 to 9.
    inline unsigned int diskNumber(align::ID, const TrackerTopology*);

    /// Endcap number is 1 at -z side and 2 at +z side.
    inline unsigned int endcapNumber(align::ID, const TrackerTopology*);
  }
}

unsigned int align::tec::moduleNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tecModule(id);
}

unsigned int align::tec::ringNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tecRing(id);
}

unsigned int align::tec::petalNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tecPetalInfo(id)[1];
}

unsigned int align::tec::sideNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tecPetalInfo(id)[0];
}

unsigned int align::tec::diskNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tecWheel(id);
}

unsigned int align::tec::endcapNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tecSide(id);
}

#endif
