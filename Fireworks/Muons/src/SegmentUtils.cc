#include "Fireworks/Muons/interface/SegmentUtils.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include <iostream>
#include <cassert>
#include <math.h>

namespace fireworks
{
  void createSegment(int detector, 
                     bool matchedSegment,
                     double segmentLength, 
                     double* segmentPosition,   
                     double* segmentDirection,
                     double* segmentInnerPoint, 
                     double* segmentOuterPoint)
  {
    if ( matchedSegment )
    {
      segmentOuterPoint[0] = segmentPosition[0] + segmentLength*segmentDirection[0];
      segmentOuterPoint[1] = segmentPosition[1] + segmentLength*segmentDirection[1];
      segmentOuterPoint[2] = segmentLength;

      segmentInnerPoint[0] = segmentPosition[0] - segmentLength*segmentDirection[0];
      segmentInnerPoint[1] = segmentPosition[1] - segmentLength*segmentDirection[1];
      segmentInnerPoint[2] = -segmentLength;
         
      return;
    }
    
    if ( detector == MuonSubdetId::DT && ! matchedSegment )
    {
   
      double mag = sqrt(segmentDirection[0]*segmentDirection[0] + 
                        segmentDirection[1]*segmentDirection[1] +
                        segmentDirection[2]*segmentDirection[2]);
      
      double theta = atan2(sqrt(segmentDirection[0]*segmentDirection[0]
                                + segmentDirection[1]*segmentDirection[1]), segmentDirection[2]);

      segmentLength /= cos(theta);

      segmentInnerPoint[0] = segmentPosition[0] + (segmentDirection[0]/mag)*segmentLength;
      segmentInnerPoint[1] = segmentPosition[1] + (segmentDirection[1]/mag)*segmentLength;
      segmentInnerPoint[2] = segmentPosition[2] + (segmentDirection[2]/mag)*segmentLength;

      segmentOuterPoint[0] = segmentPosition[0] - (segmentDirection[0]/mag)*segmentLength;
      segmentOuterPoint[1] = segmentPosition[1] - (segmentDirection[1]/mag)*segmentLength;
      segmentOuterPoint[2] = segmentPosition[2] - (segmentDirection[2]/mag)*segmentLength;
      
      return;
    }
      
    if ( detector == MuonSubdetId::CSC && ! matchedSegment )
    {
      segmentOuterPoint[0] = segmentPosition[0] + segmentDirection[0]*(segmentPosition[2]/segmentDirection[2]);
      segmentOuterPoint[1] = segmentPosition[1] + segmentDirection[1]*(segmentPosition[2]/segmentDirection[2]);
      segmentOuterPoint[2] = segmentPosition[2];

      if ( fabs(segmentOuterPoint[1]) > segmentLength )
        segmentOuterPoint[1] = segmentLength*(segmentOuterPoint[1]/fabs(segmentOuterPoint[1]));

      segmentInnerPoint[0] = segmentPosition[0] - segmentDirection[0]*(segmentPosition[2]/segmentDirection[2]);
      segmentInnerPoint[1] = segmentPosition[1] - segmentDirection[1]*(segmentPosition[2]/segmentDirection[2]);
      segmentInnerPoint[2] = -segmentPosition[2];

      if ( fabs(segmentInnerPoint[1]) > segmentLength )
        segmentInnerPoint[1] = segmentLength*(segmentInnerPoint[1]/fabs(segmentInnerPoint[1]));
      
      return;
    }
    
    fwLog(fwlog::kWarning) <<" MuonSubdetId: "<< detector <<std::endl;
    return;
  }
  

}
