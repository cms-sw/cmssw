#ifndef __L1Analysis_L1AnalysisRecoClusterDataFormat_H__
#define __L1Analysis_L1AnalysisRecoClusterDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
//
// Addition of reco information
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  class L1AnalysisRecoClusterDataFormat
  {
  public:
    L1AnalysisRecoClusterDataFormat(){Reset();}; 
    ~L1AnalysisRecoClusterDataFormat(){Reset();};
    
    void Reset()
    {
     nClusters = 0;
     eta.clear();
     phi.clear();
     et.clear();
     e.clear();
    }
    
    unsigned nClusters;
    std::vector<double> eta;
    std::vector<double> phi;
    std::vector<double> et;
    std::vector<double> e;
    
  }; 
}
#endif


