#ifndef __L1Analysis_L1AnalysisRecoMuon_H__
#define __L1Analysis_L1AnalysisRecoMuon_H__

//-------------------------------------------------------------------------------
// Created 05/02/2010 - A.C. Le Bihan
// 
//
// Original code : L1TriggerDPG/L1Ntuples/L1RecoMuonProducer - Luigi Guiducci
//-------------------------------------------------------------------------------

#include "FWCore/Framework/interface/Event.h"
#include "L1AnalysisRecoMuonDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisRecoMuon
  {
  public:
    L1AnalysisRecoMuon();
    ~L1AnalysisRecoMuon();
    
    void Reset() {recoMuon_.Reset();}
    //void Print(std::ostream &os = std::cout) const;
    void Set();
    L1AnalysisRecoMuonDataFormat * getData() {return &recoMuon_;}

  private :
    L1AnalysisRecoMuonDataFormat recoMuon_;
  }; 
}
#endif



