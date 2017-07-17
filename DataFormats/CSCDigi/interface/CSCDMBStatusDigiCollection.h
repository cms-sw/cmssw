#ifndef CSCDMBStatusDigi_CSCDMBStatusDigiCollection_h
#define CSCDMBStatusDigi_CSCDMBStatusDigiCollection_h

/** \class CSCDMBStatusDigiCollection
 *  Alex Tumanov 5/16/07
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCDMBStatusDigi> CSCDMBStatusDigiCollection;

#endif
