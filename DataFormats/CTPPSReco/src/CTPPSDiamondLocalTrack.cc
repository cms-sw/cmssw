/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *
 ****************************************************************************/

#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

//----------------------------------------------------------------------------------------------------

bool
operator<( const CTPPSDiamondLocalTrack& lhs, const CTPPSDiamondLocalTrack& rhs )
{
  // as for now, only sort by space coordinate
  return ( lhs.getT() < rhs.getT() );
}
