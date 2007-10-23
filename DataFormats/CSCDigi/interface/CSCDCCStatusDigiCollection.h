#ifndef CSCDCCStatusDigi_CSCDCCStatusDigiCollection_h
#define CSCDCCStatusDigi_CSCDCCStatusDigiCollection_h

/** \class CSCDCCStatusDigiCollection
 *
 *  $Date: 2007/03/29 16:04:42 $
 *  $Revision: 1.1 $
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCDigi/interface/CSCDCCStatusDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<CSCDetId, CSCDCCStatusDigi> CSCDCCStatusDigiCollection;

#endif
