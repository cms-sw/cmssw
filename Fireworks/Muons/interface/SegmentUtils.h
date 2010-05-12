#ifndef FIREWORKS_MUONS_SEGMENTUTILS_H
#define FIREWORKS_MUONS_SEGMENTUTILS_H

namespace fireworks
{
  void createSegment(int detector, bool matchedSegment, double segmentLength, 
                     double* segmentPosition,   double* segmentDirection,
                     double* segmentInnerPoint, double* segmentOuterPoint);
}

#endif
