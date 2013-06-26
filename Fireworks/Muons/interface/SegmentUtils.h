#ifndef FIREWORKS_MUONS_SEGMENTUTILS_H
#define FIREWORKS_MUONS_SEGMENTUTILS_H

namespace fireworks
{
  void createSegment( int detector,                // DT,CSC, or RPC? 
		      bool matchedSegment,         // Is the segment a MuonSegmentMatch?
		      float segmentLength,         // Nominal length of the segment along chamber thickness
		      float segmentLimit,          // Limit of the segment extent (i.e. stay inside chamber along a certain dimension) 
		      float* segmentPosition,      // Segment position in local coordinates 
		      float* segmentDirection,     // Segment direction in local coordinates
		      float* segmentInnerPoint,    // Line set connect this point...
		      float* segmentOuterPoint );  // ...with this one
}

#endif
