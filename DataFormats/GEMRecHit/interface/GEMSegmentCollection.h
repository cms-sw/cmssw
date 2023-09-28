#ifndef DataFormats_GEMSegmentCollection_H
#define DataFormats_GEMSegmentCollection_H

/** \class GEMSegmentCollection
 *
 * The collection of GEMSegment's. See \ref CSCSegmentCollection.h for details from which is derived.
 *
 *  \author Piet Verwilligen
 */
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMRecHit/interface/GEMSegment.h"

#include "DataFormats/Common/interface/IdToHitRange.h"

using GEMSegmentCollection = edm::IdToHitRange<GEMDetId, GEMSegment>;

#include "DataFormats/Common/interface/Ref.h"
using GEMSegmentRef = edm::Ref<GEMSegmentCollection>;

#endif
