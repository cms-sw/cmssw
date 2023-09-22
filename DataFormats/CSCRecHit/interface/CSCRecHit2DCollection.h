#ifndef DataFormats_CSCRecHit2DCollection_H
#define DataFormats_CSCRecHit2DCollection_H

/** \class CSCRecHit2DCollection
 *
 * The collection of CSCRecHit2D's. See \ref CSCRecHit2DCollection.h for details.
 *
 */
#include <vector>

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"

#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"

using CSCRecHit2DCollection = edm::RangeMap<CSCDetId, std::vector<CSCRecHit2D> >;

#endif
