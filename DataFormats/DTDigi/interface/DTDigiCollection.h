#ifndef DTDigi_DTDigiCollection_h
#define DTDigi_DTDigiCollection_h

/** \class DTDigiCollection
 *  ED
 *
 *  $Date: 2005/10/07 17:40:53 $
 *  $Revision: 1.1 $
 *  \author Stefano ARGIRO
 */

/** \class DTDigiCollection
   
   \author Stefano ARGIRO
   \version $Id: DTDigiCollection.h,v 1.1 2005/10/07 17:40:53 namapane Exp $
   \date 21 Apr 2005
*/

#include <DataFormats/MuonDetId/interface/DTLayerId.h>
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<DTLayerId, DTDigi> DTDigiCollection;

#endif

