#ifndef DTDigi_DTDigiCollection_h
#define DTDigi_DTDigiCollection_h

/** \class DTDigiCollection
 *  The collection containing DT Digis in the event.
 *  Digis are grouped by DTLayerId.
 *
 *  $Date: 2006/01/20 14:58:16 $
 *  $Revision: 1.2 $
 *  \author Stefano ARGIRO
 */

#include <DataFormats/MuonDetId/interface/DTLayerId.h>
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<DTLayerId, DTDigi> DTDigiCollection;

#endif

