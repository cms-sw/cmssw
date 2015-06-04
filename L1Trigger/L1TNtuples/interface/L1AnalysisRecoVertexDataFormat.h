#ifndef __L1Analysis_L1AnalysisRecoVertexDataFormat_H__
#define __L1Analysis_L1AnalysisRecoVertexDataFormat_H__

//-------------------------------------------------------------------------------
// Created 15/04/2010 - E. Conte, A.C. Le Bihan
// 
//
// Original code : L1Trigger/L1TNtuples/L1TrackVertexRecoTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisRecoVertexDataFormat
  {
    L1AnalysisRecoVertexDataFormat(){};
    ~L1AnalysisRecoVertexDataFormat(){};
    
    void Reset()
    {
     nVtx = 0;
     NDoF.clear();
     Z.clear();
     Rho.clear();
    }
           
    unsigned nVtx;
    std::vector<unsigned int> NDoF;
    std::vector<double>       Z;
    std::vector<double>       Rho;
    
  }; 
}
#endif


