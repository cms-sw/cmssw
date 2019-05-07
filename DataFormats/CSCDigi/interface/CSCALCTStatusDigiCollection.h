#ifndef CSCALCTStatusDigi_CSCALCTStatusDigiCollection_h
#define CSCALCTStatusDigi_CSCALCTStatusDigiCollection_h

/** \class CSCALCTStatusDigiCollection
 *
 */

#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

typedef MuonDigiCollection<CSCDetId, CSCALCTStatusDigi>
    CSCALCTStatusDigiCollection;

#endif
