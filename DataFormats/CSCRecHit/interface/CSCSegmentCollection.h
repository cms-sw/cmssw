#ifndef DataFormats_CSCSegmentCollection_H
#define DataFormats_CSCSegmentCollection_H

/** \class CSCSegmentCollection
 *
 * The collection of CSCSegment's. See \ref CSCSegmentCollection.h for details.
 *
 *  \author Matteo Sani
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"

#include "DataFormats/Common/interface/IdToHitRange.h"

using CSCSegmentCollection = edm::IdToHitRange<CSCDetId, CSCSegment>;

#include "DataFormats/Common/interface/Ref.h"
using CSCSegmentRef = edm::Ref<CSCSegmentCollection>;

//typedef std::vector<CSCSegment> CSCSegmentCollection;

#endif
