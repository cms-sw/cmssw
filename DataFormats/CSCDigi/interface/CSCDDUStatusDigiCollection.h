#ifndef CSCDDUStatusDigi_CSCDDUStatusDigiCollection_h
#define CSCDDUStatusDigi_CSCDDUStatusDigiCollection_h

/** \class CSCDDUStatusDigiCollection
 *
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCDDUStatusDigi> CSCDDUStatusDigiCollection;

#endif
