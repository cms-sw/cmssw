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
	 
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"

typedef edm::RangeMap <GEMDetId, edm::OwnVector<GEMSegment> > GEMSegmentCollection;

#include "DataFormats/Common/interface/Ref.h"
typedef edm::Ref<GEMSegmentCollection> GEMSegmentRef;
	
#endif
