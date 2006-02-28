#ifndef DataFormats_CSCRecHitCollection_H
#define DataFormats_CSCRecHitCollection_H

/** \class CSCRecHitCollection
 *
 * The collection of CSCRecHit2D's.
 *
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"

typedef MuonDigiCollection<CSCDetId, CSCRecHit2D> CSCRecHitCollection;

#endif
