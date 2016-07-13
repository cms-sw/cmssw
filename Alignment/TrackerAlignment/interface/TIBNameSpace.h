#ifndef Alignment_CommonAlignment_TIBNameSpace_H
#define Alignment_CommonAlignment_TIBNameSpace_H

/** \namespace tib
 *
 *  Namespace for numbering components in Tracker Inner Barrel.
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
  namespace tib
  {
    /// Number of strings for each surface of a half shell.
    std::vector<unsigned int> sphs;

    /// Module number increases with |z| from 1 to 3.
    inline unsigned int moduleNumber(align::ID, const TrackerTopology*);

    /// String number increases with |phi| from right (1) to left (sphs)
    /// of each half shell.
    inline unsigned int stringNumber(align::ID, const TrackerTopology*);

    /// Surface number is 1 for inner and 2 for outer.
    inline unsigned int surfaceNumber(align::ID, const TrackerTopology*);

    /// Half shell number is 1 for bottom (-y) and 2 for top (+y). 
    inline unsigned int halfShellNumber(align::ID, const TrackerTopology*);

    /// Layer number increases with rho from 1 to 8.
    inline unsigned int layerNumber(align::ID, const TrackerTopology*);

    /// Half barrel number is 1 at -z side and 2 at +z side.
    inline unsigned int halfBarrelNumber(align::ID, const TrackerTopology*);

    /// Barrel number is 1 for all align::ID's which belong to this barrel
    inline unsigned int barrelNumber(align::ID, const TrackerTopology*);
  }
}

unsigned int align::tib::moduleNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tibModule(id);
}

unsigned int align::tib::stringNumber(align::ID id, const TrackerTopology* tTopo)
{
  

  std::vector<unsigned int> s = tTopo->tibStringInfo(id);
  // s[1]: surface lower = 1, upper = 2
  // s[2]: string no. increases with phi

  unsigned int l = 2 * (tTopo->tibLayer(id) - 1) + s[1] - 1;

// String on +y surface: number = s                (1 to sphs)
// String in -y surface: number = 2 * sphs + 1 - s (1 to sphs)

  return s[2] > sphs[l] ? 2 * sphs[l] + 1 - s[2] : s[2];
}

unsigned int align::tib::surfaceNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tibStringInfo(id)[1];
}

unsigned int align::tib::halfShellNumber(align::ID id, const TrackerTopology* tTopo)
{
  

  std::vector<unsigned int> s = tTopo->tibStringInfo(id);
  // s[1]: surface lower = 1, upper = 2
  // s[2]: string no. increases with phi

  unsigned int l = 2 * (tTopo->tibLayer(id) - 1) + s[1] - 1;

  return s[2] > sphs[l] ? 1 : 2;
}

unsigned int align::tib::layerNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tibLayer(id);
}

unsigned int align::tib::halfBarrelNumber(align::ID id, const TrackerTopology* tTopo)
{
  return tTopo->tibStringInfo(id)[0];
}

unsigned int align::tib::barrelNumber(align::ID, const TrackerTopology*)
{
  return 1;
}

#endif
