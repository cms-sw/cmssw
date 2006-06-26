#ifndef DataFormats_CSCSegmentCollection_H
#define DataFormats_CSCSegmentCollection_H

/** \class CSCSegmentCollection
 *
 * The collection of CSCSegment's. See \ref CSCSegmentCollection.h for details.
 *
 *  $Date: 2006/06/14 12:08:04 $
 *  \author Matteo Sani
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h> 
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>

#include <DataFormats/Common/interface/RangeMap.h>
#include <DataFormats/Common/interface/ClonePolicy.h>
#include <DataFormats/Common/interface/OwnVector.h>

typedef edm::RangeMap <CSCDetId, edm::OwnVector<CSCSegment> > CSCSegmentCollection;

//typedef std::vector<CSCSegment> CSCSegmentCollection; 
	
#endif
