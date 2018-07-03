#ifndef __L1Analysis_L1AnalysisRecoRpcHitDataFormat_H__
#define __L1Analysis_L1AnalysisRecoRpcHitDataFormat_H__

//-------------------------------------------------------------------------------
// Created 21/11/2012 - C. Battilana
// 
//
// Original code : L1Trigger/L1TNtuples/L1RecoMuonProducer - Luigi Guiducci
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisRecoRpcHitDataFormat
  {
    L1AnalysisRecoRpcHitDataFormat(){Reset();};
    ~L1AnalysisRecoRpcHitDataFormat(){Reset();};
    
    void Reset()
    {
      
      nRpcHits = 0;
      
      region.clear();
      clusterSize.clear();
      strip.clear();
      bx.clear();
 
      xLoc.clear();
      phiGlob.clear();
      
      station.clear();
      sector.clear();
      layer.clear();
      subsector.clear();
      roll.clear();
      ring.clear();
      muonId.clear();
      
    }

    int nRpcHits;
    
    std::vector<int> region;
    std::vector<int> clusterSize;
    std::vector<int> strip;
    std::vector<int> bx;
 
    std::vector<float> xLoc;
    std::vector<float> phiGlob;
    
    std::vector<int> station;
    std::vector<int> sector;
    std::vector<int> layer;
    std::vector<int> subsector;
    std::vector<int> roll;
    std::vector<int> ring;
    std::vector<int> muonId;
    
  };
}
#endif


