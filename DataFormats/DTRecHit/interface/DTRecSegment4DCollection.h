#ifndef DTRecSegment4DCollection_H
#define DTRecSegment4DCollection_H

/** \class DTRecSegment4DCollection
 *  
 *  Collection of DTRecSegment4D. See \ref DTRecSegment4DCollection.h for details
 *
 *  \author R. Bellan - INFN Torino
 */

/* Base Class Headers */
#include <functional>
#include <vector>

/* Collaborating Class Declarations */
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

using DTRecSegment4DCollection = edm::RangeMap<DTChamberId, std::vector<DTRecSegment4D>>;

#include "DataFormats/Common/interface/Ref.h"
using DTRecSegment4DRef = edm::Ref<DTRecSegment4DCollection>;

#endif
