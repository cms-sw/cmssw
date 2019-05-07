#ifndef CSCComparatorDigi_CSCComparatorDigiCollection_h
#define CSCComparatorDigi_CSCComparatorDigiCollection_h

/** \class CSCComparatorDigiCollection
 *  ED
 *
 *  \author Michael Schmitt, Northwestern
 */

#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

typedef MuonDigiCollection<CSCDetId, CSCComparatorDigi>
    CSCComparatorDigiCollection;

#endif
