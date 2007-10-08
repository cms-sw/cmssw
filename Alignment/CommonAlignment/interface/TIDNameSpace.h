#ifndef Alignment_CommonAlignment_TIDNameSpace_H
#define Alignment_CommonAlignment_TIDNameSpace_H

/** \namespace tid
 *
 *  Namespace for numbering components in Tracker Inner Disks.
 *
 *  A system to number a component within its parent; starts from 1.
 *
 *  $Date: 2007/04/09 00:40:21 $
 *  $Revision: 1.7 $
 *  \author Chung Khim Lae
 */

#include "CondFormats/Alignment/interface/Definitions.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

namespace align
{
  namespace tid
  {
     /// Module number increases with phi.
    inline unsigned int moduleNumber( align::ID );

    /// Side number is 1 for back ring and 2 for front (towards IP).
    inline unsigned int sideNumber( align::ID );

    /// Ring number increases with rho from 1 to 3.
    inline unsigned int ringNumber( align::ID );

    /// Disk number increases with |z| from 1 to 3.
    inline unsigned int diskNumber( align::ID );

    /// Endcap number is 1 at -z side and 2 at +z side.
    inline unsigned int endcapNumber( align::ID );
  }
}

unsigned int align::tid::moduleNumber(align::ID id)
{
  return TIDDetId(id).module()[1];
}

unsigned int align::tid::sideNumber(align::ID id)
{
  return TIDDetId(id).module()[0];
}

unsigned int align::tid::ringNumber(align::ID id)
{
  return TIDDetId(id).ring();
}

unsigned int align::tid::diskNumber(align::ID id)
{
  return TIDDetId(id).wheel();
}

unsigned int align::tid::endcapNumber(align::ID id)
{
  return TIDDetId(id).side();
}

#endif
