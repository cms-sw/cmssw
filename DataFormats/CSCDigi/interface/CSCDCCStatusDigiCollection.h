#ifndef CSCDCCStatusDigi_CSCDCCStatusDigiCollection_h
#define CSCDCCStatusDigi_CSCDCCStatusDigiCollection_h

/** \class CSCDCCStatusDigiCollection
 *
 */

#include "DataFormats/CSCDigi/interface/CSCDCCStatusDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

typedef MuonDigiCollection<CSCDetId, CSCDCCStatusDigi>
    CSCDCCStatusDigiCollection;

#endif
