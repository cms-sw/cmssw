#ifndef CSCDCCFormatStatusDigi_CSCDCCFormatStatusDigiCollection_h
#define CSCDCCFormatStatusDigi_CSCDCCFormatStatusDigiCollection_h

/** \class CSCDCCFormatStatusDigiCollection
 *
 *  $Date:$
 *  $Revision:$
 *  \author N. Terentiev, CMU
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<CSCDetId, CSCDCCFormatStatusDigi> CSCDCCFormatStatusDigiCollection;

#endif
