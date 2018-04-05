#ifndef __L1Analysis_L1AnalysisL1CaloTowerDataFormat_H__
#define __L1Analysis_L1AnalysisL1CaloTowerDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisL1CaloTowerDataFormat
  {
    L1AnalysisL1CaloTowerDataFormat(){Reset();};
    ~L1AnalysisL1CaloTowerDataFormat(){};
    
    
    void Reset() {
      nTower = 0;
      ieta.clear();
      iphi.clear();
      iet.clear();
      iem.clear();
      ihad.clear();
      iratio.clear();
      iqual.clear();
      et.clear();
      eta.clear();
      phi.clear();
    }
    
    void Init() {

    }
    

    short nTower;
    std::vector<short> ieta;
    std::vector<short> iphi;
    std::vector<short> iet;
    std::vector<short> iem;
    std::vector<short> ihad;
    std::vector<short> iratio;
    std::vector<short> iqual;
    std::vector<float> et;
    std::vector<float> eta;
    std::vector<float> phi;
    
  }; 
} 
#endif

