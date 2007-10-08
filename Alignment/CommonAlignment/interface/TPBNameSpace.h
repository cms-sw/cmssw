#ifndef Alignment_CommonAlignment_TPBNameSpace_H
#define Alignment_CommonAlignment_TPBNameSpace_H

/** \namespace tpb
 *
 *  Namespace for numbering components in Barrel Pixel.
 *
 *  A system to number a component within its parent; starts from 1.
 *
 *  $Date: 2007/04/09 00:40:21 $
 *  $Revision: 1.7 $
 *  \author Chung Khim Lae
 */

#include "CondFormats/Alignment/interface/Definitions.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

namespace align
{
  namespace tpb
  {
    /// Number of ladders for each quarter cylinder.
    const unsigned int lpqc[] = {5, 8, 11};

    /// Module number increases with z from 1 to 8.
    inline unsigned int moduleNumber( align::ID );

    /// Ladder number increases from 1 at the top to 2 * lpqc at the bottom
    /// of each half cylinder.
    inline unsigned int ladderNumber( align::ID );

    /// Layer number increases with rho from 1 to 3.
    inline unsigned int layerNumber( align::ID );

    /// Half barrel number is 1 at left side (-x) and 2 at right side (+x).
    inline unsigned int halfBarrelNumber( align::ID );
  }
}

unsigned int align::tpb::moduleNumber(align::ID id)
{
  return PXBDetId(id).module();
}

unsigned int align::tpb::ladderNumber(align::ID id)
{
  PXBDetId detId(id);

  unsigned int l = detId.ladder(); // increases with phi
  unsigned int c = detId.layer() - 1;

// Ladder in 1st quadrant: number = lpqc + 1 - l     (1 to lpqc)
// Ladder in 2nd quadrant: number = l - lpqc         (1 to lpqc)
// Ladder in 3rd quadrant: number = l - lpqc         (lpqc + 1 to 2 * lpqc)
// Ladder in 4th quadrant: number = 5 * lpqc + 1 - l (lpqc + 1 to 2 * lpqc)

  return l > 3 * lpqc[c] ? // ladder in 4th quadrant
    5 * lpqc[c] + 1 - l :
    (l > lpqc[c] ? // ladder not in 1st quadrant
     l - lpqc[c] : lpqc[c] + 1 - l);
}

unsigned int align::tpb::layerNumber(align::ID id)
{
  return PXBDetId(id).layer();
}

unsigned int align::tpb::halfBarrelNumber(align::ID id)
{
  PXBDetId detId(id);

  unsigned int l = detId.ladder(); // increases with phi
  unsigned int c = detId.layer() - 1;

  return l > lpqc[c] && l <= 3 * lpqc[c] ? 1 : 2;
}

#endif
