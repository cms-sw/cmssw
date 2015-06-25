#ifndef __L1Analysis_L1AnalysisRecoMetDataFormat_H__
#define __L1Analysis_L1AnalysisRecoMetDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
//
// Addition of met reco information
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisRecoMetDataFormat
  {
    L1AnalysisRecoMetDataFormat(){Reset();};
    ~L1AnalysisRecoMetDataFormat(){Reset();};
    
    void Reset()
    {
     met    = -999;
     metPhi = -999;
     Ht     = -999;
     mHt    = -999;
     mHtPhi = -999;
     sumEt  = -999;
     ecalFlag = 0;
     hcalFlag = 0;
    }
    
    double met;
    double metPhi;
    double Ht;
    double mHt;
    double mHtPhi;
    double sumEt;
    unsigned ecalFlag;
    unsigned hcalFlag;
    
  }; 
}
#endif


