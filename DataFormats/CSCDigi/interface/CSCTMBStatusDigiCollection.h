#ifndef CSCTMBStatusDigi_CSCTMBStatusDigiCollection_h
#define CSCTMBStatusDigi_CSCTMBStatusDigiCollection_h

/** \class CSCTMBStatusDigiCollection
 *
 *  $Date: 2007/03/29 16:04:42 $
 *  $Revision: 1.1 $
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<CSCDetId, CSCTMBStatusDigi> CSCTMBStatusDigiCollection;

#endif
