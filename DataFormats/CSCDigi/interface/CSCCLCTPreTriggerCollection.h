#ifndef CSCStripDigi_CSCCLCTPreTriggerCollection_h
#define CSCStripDigi_CSCCLCTPreTriggerCollection_h

/** \class CSCCLCTPreTriggerCollection
 *  ED
 *
 *  $Date: 2010/04/23 20:39:16 $
 *  $Revision: 1.1 $
 *  \author Rick Wilkinson
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
// just the BX for now
typedef int CSCCLCTPreTrigger;
typedef MuonDigiCollection<CSCDetId, CSCCLCTPreTrigger> CSCCLCTPreTriggerCollection;

#endif

