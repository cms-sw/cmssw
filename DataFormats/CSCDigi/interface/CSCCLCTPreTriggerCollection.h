#ifndef CSCStripDigi_CSCCLCTPreTriggerCollection_h
#define CSCStripDigi_CSCCLCTPreTriggerCollection_h

/** \class CSCCLCTPreTriggerCollection
 *  ED
 *
 *  \author Rick Wilkinson
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
// just the BX for now
typedef int CSCCLCTPreTrigger;
typedef MuonDigiCollection<CSCDetId, CSCCLCTPreTrigger> CSCCLCTPreTriggerCollection;

#endif
