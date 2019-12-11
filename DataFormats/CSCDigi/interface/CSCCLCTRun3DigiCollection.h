#ifndef DataFormats_CSCCLCTDigi_CSCCLCTRun3DigiCollection_h
#define DataFormats_CSCCLCTDigi_CSCCLCTRun3DigiCollection_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTRun3Digi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCCLCTRun3Digi> CSCCLCTRun3DigiCollection;

#endif
