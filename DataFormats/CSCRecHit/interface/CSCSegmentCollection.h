#ifndef DataFormats_CSCSegmentCollection_H
#define DataFormats_CSCSegmentCollection_H

/** \class CSCSegmentCollection
 *
 * The collection of CSCSegment's. See \ref CSCSegmentCollection.h for details.
 *
 *  $Date: 2010/03/12 13:08:15 $
 *  \author Matteo Sani
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h> 
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>

#include <DataFormats/Common/interface/RangeMap.h>
#include <DataFormats/Common/interface/ClonePolicy.h>
#include <DataFormats/Common/interface/OwnVector.h>

typedef edm::RangeMap <CSCDetId, edm::OwnVector<CSCSegment> > CSCSegmentCollection;

#include <DataFormats/Common/interface/Ref.h>
typedef edm::Ref<CSCSegmentCollection> CSCSegmentRef;

//typedef std::vector<CSCSegment> CSCSegmentCollection; 
	
#endif
