#ifndef Alignment_CommonAlignment_TOBNameSpace_H
#define Alignment_CommonAlignment_TOBNameSpace_H

/** \namespace tob
 *
 *  Namespace for numbering components in Tracker Outer Barrel.
 *
 *  A system to number a component within its parent; starts from 1.
 *
 *  $Date: 2007/10/18 09:57:10 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include "CondFormats/Alignment/interface/Definitions.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

namespace align
{
  namespace tob
  {
     /// Module number increases with |z| from 1 to 6.
    inline unsigned int moduleNumber( align::ID );

    /// Rod number increases with phi.
    inline unsigned int rodNumber( align::ID );

    /// Layer number increases with rho from 1 to 6.
    inline unsigned int layerNumber( align::ID );

    /// HalfBarrel number is 1 at -z side and 2 at +z side.
    inline unsigned int halfBarrelNumber( align::ID );
  }
}

unsigned int align::tob::moduleNumber(align::ID id)
{
  return TOBDetId(id).module();
}

unsigned int align::tob::rodNumber(align::ID id)
{
  return TOBDetId(id).rod()[1];
}

unsigned int align::tob::layerNumber(align::ID id)
{
  return TOBDetId(id).layer();
}

unsigned int align::tob::halfBarrelNumber(align::ID id)
{
  return TOBDetId(id).rod()[0];
}

#endif
