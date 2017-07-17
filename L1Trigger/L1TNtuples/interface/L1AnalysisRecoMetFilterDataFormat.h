#ifndef __L1Analysis_L1AnalysisRecoMetFilterDataFormat_H__
#define __L1Analysis_L1AnalysisRecoMetFilterDataFormat_H__

//-------------------------------------------------------------------------------
// Created 27/01/2016 - A. Bundock
// 
//
// Addition of met reco information
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisRecoMetFilterDataFormat
  {
    L1AnalysisRecoMetFilterDataFormat(){Reset();};
    ~L1AnalysisRecoMetFilterDataFormat(){Reset();};
    
    void Reset()
    {
     
      hbheNoiseFilter         = 0;
      hbheNoiseIsoFilter      = 0;
      cscTightHalo2015Filter  = 0;
      ecalDeadCellTPFilter    = 0;
      goodVerticesFilter      = 0;
      eeBadScFilter           = 0;
      chHadTrackResFilter     = 0;
      muonBadTrackFilter      = 0;

    }
    
    bool hbheNoiseFilter;        
    bool hbheNoiseIsoFilter;     
    bool cscTightHalo2015Filter; 
    bool ecalDeadCellTPFilter;
    bool goodVerticesFilter;  
    bool eeBadScFilter;    
    bool chHadTrackResFilter;   
    bool muonBadTrackFilter;   
    
  }; 
}
#endif


