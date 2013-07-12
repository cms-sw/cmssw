#ifndef CSCStripDigi_CSCStripDigiCollection_h
#define CSCStripDigi_CSCStripDigiCollection_h

/** \class CSCStripDigiCollection
 *  ED
 *
 *  $Date: 2005/10/07 17:40:53 $
 *  $Revision: 1.1 $
 *  \author Michael Schmitt, Northwestern
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<CSCDetId, CSCStripDigi> CSCStripDigiCollection;

#endif

