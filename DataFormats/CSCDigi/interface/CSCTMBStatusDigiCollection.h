#ifndef CSCTMBStatusDigi_CSCTMBStatusDigiCollection_h
#define CSCTMBStatusDigi_CSCTMBStatusDigiCollection_h

/** \class CSCTMBStatusDigiCollection
 *
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCTMBStatusDigi> CSCTMBStatusDigiCollection;

#endif
