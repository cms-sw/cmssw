#ifndef DTDigi_DTDigiCollection_h
#define DTDigi_DTDigiCollection_h

/** \class DTDigiCollection
 *  The collection containing DT Digis in the event.
 *  Digis are grouped by DTLayerId.
 *
 *  \author Stefano ARGIRO
 */

#include <DataFormats/MuonDetId/interface/DTLayerId.h>
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<DTLayerId, DTDigi> DTDigiCollection;

#endif
