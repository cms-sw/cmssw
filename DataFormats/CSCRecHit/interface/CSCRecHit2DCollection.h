#ifndef DataFormats_CSCRecHit2DCollection_H
#define DataFormats_CSCRecHit2DCollection_H


/** \class CSCRecHit2DCollection
 *
 * The collection of CSCRecHit2D's.
 *
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"

typedef MuonDigiCollection<CSCDetId, CSCRecHit2D> CSCRecHit2DCollection;

#endif
