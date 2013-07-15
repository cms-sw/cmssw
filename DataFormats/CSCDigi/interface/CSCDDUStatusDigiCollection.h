#ifndef CSCDDUStatusDigi_CSCDDUStatusDigiCollection_h
#define CSCDDUStatusDigi_CSCDDUStatusDigiCollection_h

/** \class CSCDDUStatusDigiCollection
 *
 *  $Date: 2007/03/29 16:04:42 $
 *  $Revision: 1.1 $
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<CSCDetId, CSCDDUStatusDigi> CSCDDUStatusDigiCollection;

#endif
