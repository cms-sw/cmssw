#ifndef __L1Analysis_L1AnalysisGT_H__
#define __L1Analysis_L1AnalysisGT_H__

//-------------------------------------------------------------------------------
// Created 06/01/2010 - A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------


#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "L1AnalysisGTDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisGT 
  {
  public:
    L1AnalysisGT();
    ~L1AnalysisGT();
    
    void Set(const L1GlobalTriggerReadoutRecord* gtrr);
    void SetEvm(const L1GlobalTriggerEvmReadoutRecord* gtevmrr);
    void Reset() {gt_.Reset();}
    L1AnalysisGTDataFormat* getData() {return &gt_;}
   
  private :
    L1AnalysisGTDataFormat gt_;
  }; 
} 
#endif


