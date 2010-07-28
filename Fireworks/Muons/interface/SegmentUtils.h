#ifndef FIREWORKS_MUONS_SEGMENTUTILS_H
#define FIREWORKS_MUONS_SEGMENTUTILS_H

class TGeoHMatrix;
class TEveStraightLineSet;

namespace fireworks
{
  void createSegment( int detector,                // DT,CSC, or RPC? 
		      bool matchedSegment,         // Is the segment a MuonSegmentMatch?
		      double segmentLength,        // Nominal length of the segment along chamber thickness
		      double segmentLimit,         // Limit of the segment extent (i.e. stay inside chamber along a certain dimension) 
		      double* segmentPosition,     // Segment position in local coordinates 
		      double* segmentDirection,    // Segment direction in local coordinates
		      double* segmentInnerPoint,   // Line set connect this point...
		      double* segmentOuterPoint ); // ...with this one
}

#endif
