#ifndef CSCCorrelatedLCTDigi_CSCCorrelatedLCTDigiCollection_h
#define CSCCorrelatedLCTDigi_CSCCorrelatedLCTDigiCollection_h

/** \class CSCCorrelatedLCTDigiCollection
 *
 *  Based on DTDigiCollection.h
 *  \author L. Gray - UF
 *
 */

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

typedef MuonDigiCollection<CSCDetId, CSCCorrelatedLCTDigi>
    CSCCorrelatedLCTDigiCollection;

#endif
