#ifndef __L1Analysis_L1AnalysisL1MenuDataFormat_H__
#define __L1Analysis_L1AnalysisL1MenuDataFormat_H__

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisL1MenuDataFormat
  {
    L1AnalysisL1MenuDataFormat() {Reset();}
    ~L1AnalysisL1MenuDataFormat() {}
    
    void Reset()
    {
      AlgoTrig_PrescaleFactorIndexValid = false;
      AlgoTrig_PrescaleFactorIndex = 0;
      TechTrig_PrescaleFactorIndexValid = false;
      TechTrig_PrescaleFactorIndex = 0;
    }

    bool AlgoTrig_PrescaleFactorIndexValid;
    int  AlgoTrig_PrescaleFactorIndex;
    bool TechTrig_PrescaleFactorIndexValid;
    int  TechTrig_PrescaleFactorIndex;



  }; 
}
#endif


