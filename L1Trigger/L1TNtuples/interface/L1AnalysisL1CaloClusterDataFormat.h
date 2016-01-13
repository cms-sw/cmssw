#ifndef __L1Analysis_L1AnalysisL1CaloClusterDataFormat_H__
#define __L1Analysis_L1AnalysisL1CaloClusterDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisL1CaloClusterDataFormat
  {
    L1AnalysisL1CaloClusterDataFormat(){Reset();};
    ~L1AnalysisL1CaloClusterDataFormat(){};
    
    
    void Reset() {
      nCluster = 0;
      ieta.clear();
      iphi.clear();
      iet.clear();
      iqual.clear();
      et.clear();
      eta.clear();
      phi.clear();
    }
    
    void Init() {

    }
    

    short int nCluster;
    std::vector<short int> ieta;
    std::vector<short int> iphi;
    std::vector<short int> iet;
    std::vector<short int> iqual;
    std::vector<float> et;
    std::vector<float> eta;
    std::vector<float> phi;
    
  }; 
} 
#endif

