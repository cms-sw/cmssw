#ifndef DataFormats_ME0SegmentCollection_H
#define DataFormats_ME0SegmentCollection_H

/** \class ME0SegmentCollection
 *
 * The collection of ME0Segment's. See \ref CSCSegmentCollection.h for details from which is derived.
 *
 *  $Date: 2014/02/04 10:08:15 $
 *  \author Marcello Maggi
 */

#include <DataFormats/MuonDetId/interface/ME0DetId.h> 
#include <DataFormats/GEMRecHit/interface/ME0Segment.h>

#include <DataFormats/Common/interface/RangeMap.h>
#include <DataFormats/Common/interface/ClonePolicy.h>
#include <DataFormats/Common/interface/OwnVector.h>

typedef edm::RangeMap <ME0DetId, edm::OwnVector<ME0Segment> > ME0SegmentCollection;

#include <DataFormats/Common/interface/Ref.h>
typedef edm::Ref<ME0SegmentCollection> ME0SegmentRef;
	
#endif
