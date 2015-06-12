
#ifndef DataFormats_EmulatedME0SegmentCollection_H
#define DataFormats_EmulatedME0SegmentCollection_H

/** \class EmulatedME0SegmentCollection
 *
 * The collection of EmulatedME0Segment's. See \ref EmulatedME0SegmentCollection.h for details.
 *
 *  $Date: 2010/03/12 13:08:15 $
 *  \author Matteo Sani
 */

#include <DataFormats/MuonReco/interface/EmulatedME0Segment.h>
#include <DataFormats/Common/interface/Ref.h>

/// collection of EmulatedME0Segments
typedef std::vector<EmulatedME0Segment> EmulatedME0SegmentCollection;

/// persistent reference to a EmulatedME0Segment
typedef edm::Ref<EmulatedME0SegmentCollection> EmulatedME0SegmentRef;

#endif


