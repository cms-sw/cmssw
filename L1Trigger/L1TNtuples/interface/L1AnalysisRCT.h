#ifndef __L1Analysis_L1AnalysisRCT_H__
#define __L1Analysis_L1AnalysisRCT_H__

//-------------------------------------------------------------------------------
// Created 06/01/2010 - A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------


#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/Common/interface/Handle.h"
#include "L1AnalysisRCTDataFormat.h" 

namespace L1Analysis
{
  class L1AnalysisRCT
  {
  public:
    L1AnalysisRCT();
    L1AnalysisRCT(int maxRCTREG);
    ~L1AnalysisRCT();
    
    void SetEmRCT(const edm::Handle < L1CaloEmCollection > em);
    void SetHdRCT(const edm::Handle < L1CaloRegionCollection > rgn);
    void Reset() {rct_.Reset();}
    L1AnalysisRCTDataFormat * getData() {return &rct_;}

  private :
    L1AnalysisRCTDataFormat rct_;
  }; 
} 
#endif


