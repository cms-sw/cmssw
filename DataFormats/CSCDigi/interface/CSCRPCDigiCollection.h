#ifndef CSCRPCDigi_CSCRPCDigiCollection_h
#define CSCRPCDigi_CSCRPCDigiCollection_h

/** \class CSCRPCDigiCollection
 *
 *  For RPC data coming thru RAT-ALCT_DDU
 *  \author N. Terentiev - CMU
 *
*/

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCRPCDigi> CSCRPCDigiCollection;

#endif
