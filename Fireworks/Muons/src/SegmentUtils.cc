#include "Fireworks/Muons/interface/SegmentUtils.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "TEveStraightLineSet.h"
#include "TGeoMatrix.h"

#include <iostream>
#include <cassert>
#include <math.h>

namespace fireworks
{
  void
  addSegment( const TGeoHMatrix* matrix, TEveStraightLineSet& oSegmentSet, double* localSegmentCenterPoint, double* localSegmentInnerPoint, double* localSegmentOuterPoint )
  {
    double globalSegmentInnerPoint[3];
    double globalSegmentCenterPoint[3];
    double globalSegmentOuterPoint[3];

    matrix->LocalToMaster( localSegmentInnerPoint, globalSegmentInnerPoint );
    matrix->LocalToMaster( localSegmentCenterPoint, globalSegmentCenterPoint );
    matrix->LocalToMaster( localSegmentOuterPoint, globalSegmentOuterPoint );

    if( globalSegmentInnerPoint[1] * globalSegmentOuterPoint[1] > 0 )
    {
      oSegmentSet.AddLine( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
			   globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
    }
    else
    {
      if( fabs(globalSegmentInnerPoint[1]) > fabs(globalSegmentOuterPoint[1]) )
	oSegmentSet.AddLine( globalSegmentInnerPoint[0], globalSegmentInnerPoint[1], globalSegmentInnerPoint[2],
			     globalSegmentCenterPoint[0], globalSegmentCenterPoint[1], globalSegmentCenterPoint[2] );
      else
	oSegmentSet.AddLine( globalSegmentCenterPoint[0], globalSegmentCenterPoint[1], globalSegmentCenterPoint[2],
			     globalSegmentOuterPoint[0], globalSegmentOuterPoint[1], globalSegmentOuterPoint[2] );
    }
  }

  // FIXME: This should provide limits for DT segments as well.
  void createSegment( int detector, 
		      bool matchedSegment,
		      double segmentLength,
		      double segmentLimit,
		      double* segmentPosition,   
		      double* segmentDirection,
		      double* segmentInnerPoint, 
		      double* segmentOuterPoint )
  {
    if( detector == MuonSubdetId::CSC )
    {
      if( matchedSegment )
      {
        segmentOuterPoint[0] = segmentPosition[0] + segmentLength*segmentDirection[0];
        segmentOuterPoint[1] = segmentPosition[1] + segmentLength*segmentDirection[1];
        segmentOuterPoint[2] = segmentLength;

        if( fabs(segmentOuterPoint[1]) > segmentLimit )
          segmentOuterPoint[1] = segmentLimit * ( segmentOuterPoint[1] / fabs( segmentOuterPoint[1] ));

        segmentInnerPoint[0] = segmentPosition[0] - segmentLength*segmentDirection[0];
        segmentInnerPoint[1] = segmentPosition[1] - segmentLength*segmentDirection[1];
        segmentInnerPoint[2] = -segmentLength;

        if( fabs(segmentInnerPoint[1]) > segmentLimit )
          segmentInnerPoint[1] = segmentLimit * ( segmentInnerPoint[1] / fabs( segmentInnerPoint[1] ));

        return;
      }
      
      else 
      {
        segmentOuterPoint[0] = segmentPosition[0] + segmentDirection[0] * ( segmentPosition[2] / segmentDirection[2] );
        segmentOuterPoint[1] = segmentPosition[1] + segmentDirection[1] * ( segmentPosition[2] / segmentDirection[2] );
        segmentOuterPoint[2] = segmentPosition[2];

        if( fabs( segmentOuterPoint[1] ) > segmentLength )
          segmentOuterPoint[1] = segmentLength * ( segmentOuterPoint[1]/fabs( segmentOuterPoint[1] ));

        segmentInnerPoint[0] = segmentPosition[0] - segmentDirection[0] * ( segmentPosition[2] / segmentDirection[2] );
        segmentInnerPoint[1] = segmentPosition[1] - segmentDirection[1] * ( segmentPosition[2] / segmentDirection[2] );
        segmentInnerPoint[2] = -segmentPosition[2];

        if( fabs( segmentInnerPoint[1] ) > segmentLength )
          segmentInnerPoint[1] = segmentLength * ( segmentInnerPoint[1] / fabs( segmentInnerPoint[1] ));
      
        return;
      }
    }
    
    if ( detector == MuonSubdetId::DT )
    {
      if( matchedSegment )
      {
        segmentOuterPoint[0] = segmentPosition[0] + segmentLength*segmentDirection[0];
        segmentOuterPoint[1] = segmentPosition[1] + segmentLength*segmentDirection[1];
        segmentOuterPoint[2] = segmentLength;

        segmentInnerPoint[0] = segmentPosition[0] - segmentLength*segmentDirection[0];
        segmentInnerPoint[1] = segmentPosition[1] - segmentLength*segmentDirection[1];
        segmentInnerPoint[2] = -segmentLength;
         
        return;
      }
      else
      {
        double mag = sqrt( segmentDirection[0] * segmentDirection[0] + 
                           segmentDirection[1] * segmentDirection[1] +
                           segmentDirection[2] * segmentDirection[2] );
      
        double theta = atan2( sqrt( segmentDirection[0] * segmentDirection[0]
                                  + segmentDirection[1] * segmentDirection[1] ), segmentDirection[2] );

        segmentLength /= cos( theta );

        segmentInnerPoint[0] = segmentPosition[0] + ( segmentDirection[0] / mag ) * segmentLength;
        segmentInnerPoint[1] = segmentPosition[1] + ( segmentDirection[1] / mag ) * segmentLength;
        segmentInnerPoint[2] = segmentPosition[2] + ( segmentDirection[2] / mag ) * segmentLength;

	std::cout << "segmentInnerPoint[0] = " << segmentInnerPoint[0]
		  << ", segmentInnerPoint[1] = " << segmentInnerPoint[1]
		  << ", segmentInnerPoint[2] = " << segmentInnerPoint[2] << std::endl;
	
        segmentOuterPoint[0] = segmentPosition[0] - ( segmentDirection[0] / mag ) * segmentLength;
        segmentOuterPoint[1] = segmentPosition[1] - ( segmentDirection[1] / mag ) * segmentLength;
        segmentOuterPoint[2] = segmentPosition[2] - ( segmentDirection[2] / mag ) * segmentLength;

	std::cout << "segmentOuterPoint[0] = " << segmentOuterPoint[0]
		  << ", segmentOuterPoint[1] = " << segmentOuterPoint[1]
		  << ", segmentOuterPoint[2] = " << segmentOuterPoint[2] << std::endl;
      
        return;
      }  
    }
    
    fwLog( fwlog::kWarning ) << "MuonSubdetId: " << detector << std::endl;
    return;
  }
  

}
