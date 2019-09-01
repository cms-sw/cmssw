#ifndef CSCCorrelatedLCTDigi_CSCCorrelatedLCTDigiCollection_h
#define CSCCorrelatedLCTDigi_CSCCorrelatedLCTDigiCollection_h

/** \class CSCCorrelatedLCTDigiCollection
 *
 *  Based on DTDigiCollection.h
 *  \author L. Gray - UF
 *
*/

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCCorrelatedLCTDigi> CSCCorrelatedLCTDigiCollection;

#endif
