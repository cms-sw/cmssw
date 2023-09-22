#ifndef DataFormats_CSCSegmentCollection_H
#define DataFormats_CSCSegmentCollection_H

/** \class CSCSegmentCollection
 *
 * The collection of CSCSegment's. See \ref CSCSegmentCollection.h for details.
 *
 *  \author Matteo Sani
 */

#include <vector>
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"

#include "DataFormats/Common/interface/RangeMap.h"

using CSCSegmentCollection = edm::RangeMap<CSCDetId, std::vector<CSCSegment> >;

#include "DataFormats/Common/interface/Ref.h"
using CSCSegmentRef = edm::Ref<CSCSegmentCollection>;

//typedef std::vector<CSCSegment> CSCSegmentCollection;

#endif
