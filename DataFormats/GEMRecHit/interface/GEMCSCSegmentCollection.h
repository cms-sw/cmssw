#ifndef DataFormats_GEMCSCSegmentCollection_H
#define DataFormats_GEMCSCSegmentCollection_H

/** \class GEMCSCSegmentCollection
 *
 * The collection of GEMCSCSegment's. See \ref GEMCSCSegmentCollection.h for details.
 *
 *  $Date:  $
 *  \author Raffaella Radogna
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h> 
#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/GEMRecHit/interface/GEMCSCSegment.h>

#include <DataFormats/Common/interface/RangeMap.h>
#include <DataFormats/Common/interface/ClonePolicy.h>
#include <DataFormats/Common/interface/OwnVector.h>


typedef edm::RangeMap <CSCDetId, edm::OwnVector<GEMCSCSegment> > GEMCSCSegmentCollection;

#include <DataFormats/Common/interface/Ref.h>
typedef edm::Ref<GEMCSCSegmentCollection> GEMCSCSegmentRef;


#endif
