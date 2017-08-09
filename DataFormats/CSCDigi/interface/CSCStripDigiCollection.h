#ifndef CSCStripDigi_CSCStripDigiCollection_h
#define CSCStripDigi_CSCStripDigiCollection_h

/** \class CSCStripDigiCollection
 *  ED
 *
 *  \author Michael Schmitt, Northwestern
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCStripDigi> CSCStripDigiCollection;

#endif

