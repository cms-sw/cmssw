#ifndef CSCComparatorDigi_CSCComparatorDigiCollection_h
#define CSCComparatorDigi_CSCComparatorDigiCollection_h

/** \class CSCComparatorDigiCollection
 *  ED
 *
 *  $Date: 2005/11/18 10:59:01 $
 *  $Revision: 1.1 $
 *  \author Michael Schmitt, Northwestern
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCDigi/interface/CSCComparatorDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<CSCDetId, CSCComparatorDigi> CSCComparatorDigiCollection;

#endif

