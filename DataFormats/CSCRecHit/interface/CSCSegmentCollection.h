#ifndef DataFormats_CSCSegmentCollection_H
#define DataFormats_CSCSegmentCollection_H


/** \class CSCSegmentCollection
 *
 * The collection of CSCSegment's.
 *
 */
 
//#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>

//#include <DataFormats/Common/interface/RangeMap.h>
//#include <DataFormats/Common/interface/ClonePolicy.h>
//#include <DataFormats/Common/interface/OwnVector.h>

//typedef edm::RangeMap <CSCDetId,
//		       edm::OwnVector<CSCRecHit2D, edm::ClonePolicy<CSCRecHit2D> >,
//		       edm::ClonePolicy<CSCRecHit2D> > CSCRecHit2DCollection;

    typedef std::vector<CSCSegment> CSCSegmentCollection; 
	
#endif
