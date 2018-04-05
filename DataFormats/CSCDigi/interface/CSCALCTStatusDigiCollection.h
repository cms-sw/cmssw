#ifndef CSCALCTStatusDigi_CSCALCTStatusDigiCollection_h
#define CSCALCTStatusDigi_CSCALCTStatusDigiCollection_h

/** \class CSCALCTStatusDigiCollection
 *
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCALCTStatusDigi> CSCALCTStatusDigiCollection;

#endif
