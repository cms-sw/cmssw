#ifndef CSCCFEBStatusDigi_CSCCFEBStatusDigiCollection_h
#define CSCCFEBStatusDigi_CSCCFEBStatusDigiCollection_h

/** \class CSCCFEBStatusDigiCollection
 *
 *  \author N. Terentiev, CMU
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCCFEBStatusDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<CSCDetId, CSCCFEBStatusDigi> CSCCFEBStatusDigiCollection;

#endif
