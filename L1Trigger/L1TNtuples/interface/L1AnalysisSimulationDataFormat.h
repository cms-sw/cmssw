#ifndef __L1Analysis_L1AnalysisSimulationDataFormat_H__
#define __L1Analysis_L1AnalysisSimulationDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  class L1AnalysisSimulationDataFormat
  {
  public:
    L1AnalysisSimulationDataFormat(){ Reset(); };
    ~L1AnalysisSimulationDataFormat(){};
    
    void Reset()
    {
      meanInt   = -1.;
      actualInt = -1;
    }

                   
    // ---- L1AnalysisSimulationDataFormat information.
    
    
    float              meanInt;
    int              actualInt;

    
  }; 
} 
#endif


