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
    

    int nTower;
    std::vector<int> ieta;
    std::vector<int> iphi;
    std::vector<int> iet;
    std::vector<int> iem;
    std::vector<int> ihad;
    std::vector<int> iratio;
    std::vector<int> iqual;
    std::vector<double> et;
    std::vector<double> eta;
    std::vector<double> phi;
    
  }; 
} 
#endif

