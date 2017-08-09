#ifndef CSCCLCTDigi_CSCCLCTDigiCollection_h
#define CSCCLCTDigi_CSCCLCTDigiCollection_h

/** \class CSCCLCTDigiCollection
 *
 *  For CLCT trigger primitives
 *  \author N. Terentiev - CMU
 *
*/

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCCLCTDigi> CSCCLCTDigiCollection;

#endif
