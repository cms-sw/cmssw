#ifndef CSCALCTStatusDigi_CSCALCTStatusDigiCollection_h
#define CSCALCTStatusDigi_CSCALCTStatusDigiCollection_h

/** \class CSCALCTStatusDigiCollection
 *
 *  $Date: 2007/05/21 20:06:55 $
 *  $Revision: 1.1 $
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<CSCDetId, CSCALCTStatusDigi> CSCALCTStatusDigiCollection;

#endif
