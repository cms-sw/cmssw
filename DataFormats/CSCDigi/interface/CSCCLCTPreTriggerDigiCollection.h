#ifndef DataFormats_CSCDigi_CSCCLCTPreTriggerDigiCollection_h
#define DataFormats_CSCDigi_CSCCLCTPreTriggerDigiCollection_h

#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

typedef MuonDigiCollection<CSCDetId, CSCCLCTPreTriggerDigi>
    CSCCLCTPreTriggerDigiCollection;

#endif
