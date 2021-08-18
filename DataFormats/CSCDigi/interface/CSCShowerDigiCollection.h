#ifndef DataFormats_CSCDigi_CSCShowerDigiCollection_h
#define DataFormats_CSCDigi_CSCShowerDigiCollection_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCShowerDigi> CSCShowerDigiCollection;

#endif
