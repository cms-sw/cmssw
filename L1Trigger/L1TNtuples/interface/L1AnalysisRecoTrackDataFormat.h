#ifndef __L1Analysis_L1AnalysisRecoTrackDataFormat_H__
#define __L1Analysis_L1AnalysisRecoTrackDataFormat_H__

//-------------------------------------------------------------------------------
// Created 15/04/2010 - E. Conte, A.C. Le Bihan
// 
//
// Original code : L1TriggerDPG/L1Ntuples/L1TrackVertexRecoTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisRecoTrackDataFormat
  {
    L1AnalysisRecoTrackDataFormat(){Reset();};
    ~L1AnalysisRecoTrackDataFormat(){Reset();};
    
    void Reset()
    {
     nTrk = 0;
     nHighPurity = 0;
     fHighPurity = 0.;
    }
           
    unsigned int nTrk;
    unsigned int nHighPurity;
    double       fHighPurity;
  
  }; 
}
#endif


