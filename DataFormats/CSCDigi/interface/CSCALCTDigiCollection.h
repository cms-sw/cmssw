#ifndef CSCALCTDigi_CSCALCTDigiCollection_h
#define CSCALCTDigi_CSCALCTDigiCollection_h

/** \class CSCALCTDigiCollection
 *
 *  For ALCT trigger primitives
 *  \author N. Terentiev - CMU
 *
*/

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCALCTDigi> CSCALCTDigiCollection;

#endif
