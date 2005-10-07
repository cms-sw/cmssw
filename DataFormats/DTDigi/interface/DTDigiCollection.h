#ifndef DTDigi_DTDigiCollection_h
#define DTDigi_DTDigiCollection_h

/** \class DTDigiCollection
 *  ED
 *
 *  $Date: $
 *  $Revision: $
 *  \author Stefano ARGIRO
 */

/** \class DTDigiCollection
   
   \author Stefano ARGIRO
   \version $Id: DTDigiCollection.h,v 1.3 2005/08/23 09:09:41 argiro Exp $
   \date 21 Apr 2005
*/

#include <DataFormats/MuonDetId/interface/DTDetId.h>
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<DTDetId, DTDigi> DTDigiCollection;

#endif

