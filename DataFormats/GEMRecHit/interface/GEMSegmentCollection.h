#ifndef DataFormats_GEMSegmentCollection_H
#define DataFormats_GEMSegmentCollection_H

/** \class GEMSegmentCollection
 *
 * The collection of GEMSegment's. See \ref CSCSegmentCollection.h for details from which is derived.
 *
 *  \author Piet Verwilligen
 */
#include <vector>
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMRecHit/interface/GEMSegment.h"

#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"

using GEMSegmentCollection = edm::RangeMap<GEMDetId, std::vector<GEMSegment>>;

#include "DataFormats/Common/interface/Ref.h"
using GEMSegmentRef = edm::Ref<GEMSegmentCollection>;

#endif
