#ifndef Alignment_CommonAlignment_TECNameSpace_H
#define Alignment_CommonAlignment_TECNameSpace_H

/** \namespace tec
 *
 *  Namespace for numbering components in Tracker Endcaps.
 *
 *  A system to number a component within its parent; starts from 1.
 *
 *  $Date: 2007/04/09 00:40:21 $
 *  $Revision: 1.7 $
 *  \author Chung Khim Lae
 */

#include "CondFormats/Alignment/interface/Definitions.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

namespace align
{
  namespace tec
  {
     /// Module number increases (decreases) with phi for +z (-z) endcap.
    inline unsigned int moduleNumber( align::ID );

    /// Ring number increases with rho.
    inline unsigned int ringNumber( align::ID );

    /// Petal number increases with phi from 1 to 8.
    inline unsigned int petalNumber( align::ID );

    /// Side number is 1 for back disk and 2 for front (towards IP).
    inline unsigned int sideNumber( align::ID );

    /// Disk number increases with |z| from 1 to 9.
    inline unsigned int diskNumber( align::ID );

    /// Endcap number is 1 at -z side and 2 at +z side.
    inline unsigned int endcapNumber( align::ID );
  }
}

unsigned int align::tec::moduleNumber(align::ID id)
{
  return TECDetId(id).module();
}

unsigned int align::tec::ringNumber(align::ID id)
{
  return TECDetId(id).ring();
}

unsigned int align::tec::petalNumber(align::ID id)
{
  return TECDetId(id).petal()[1];
}

unsigned int align::tec::sideNumber(align::ID id)
{
  return TECDetId(id).petal()[0];
}

unsigned int align::tec::diskNumber(align::ID id)
{
  return TECDetId(id).wheel();
}

unsigned int align::tec::endcapNumber(align::ID id)
{
  return TECDetId(id).side();
}

#endif
