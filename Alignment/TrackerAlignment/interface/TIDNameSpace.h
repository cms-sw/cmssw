#ifndef Alignment_CommonAlignment_TIDNameSpace_H
#define Alignment_CommonAlignment_TIDNameSpace_H

/** \namespace tid
 *
 *  Namespace for numbering components in Tracker Inner Disks.
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
  namespace tid
  {
     /// Module number increases with phi.
    inline unsigned int moduleNumber(align::ID, const TrackerTopology*);

    /// Side number is 1 for back ring and 2 for front (towards IP).
    inline unsigned int sideNumber(align::ID, const TrackerTopology*);

    /// Ring number increases with rho from 1 to 3.
    inline unsigned int ringNumber(align::ID, const TrackerTopology*);

    /// Disk number increases with |z| from 1 to 3.
    inline unsigned int diskNumber(align::ID, const TrackerTopology*);

    /// Endcap number is 1 at -z side and 2 at +z side.
    inline unsigned int endcapNumber(align::ID, const TrackerTopology*);
  }
}

unsigned int align::tid::moduleNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tidModuleInfo(id)[1];
}

unsigned int align::tid::sideNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tidModuleInfo(id)[0];
}

unsigned int align::tid::ringNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tidRing(id);
}

unsigned int align::tid::diskNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tidWheel(id);
}

unsigned int align::tid::endcapNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tidSide(id);
}

#endif
