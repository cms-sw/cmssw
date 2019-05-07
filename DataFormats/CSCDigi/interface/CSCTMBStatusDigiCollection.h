#ifndef CSCTMBStatusDigi_CSCTMBStatusDigiCollection_h
#define CSCTMBStatusDigi_CSCTMBStatusDigiCollection_h

/** \class CSCTMBStatusDigiCollection
 *
 */

#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

typedef MuonDigiCollection<CSCDetId, CSCTMBStatusDigi>
    CSCTMBStatusDigiCollection;

#endif
