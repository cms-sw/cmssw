#ifndef CSCALCTPreTriggerDigi_CSCALCTPreTriggerDigiCollection_h
#define CSCALCTPreTriggerDigi_CSCALCTPreTriggerDigiCollection_h

/** \class CSCALCTPreTriggerDigiCollection
 *
 *  For ALCT trigger primitives
 *
*/

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCALCTPreTriggerDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCALCTPreTriggerDigi> CSCALCTPreTriggerDigiCollection;

#endif
