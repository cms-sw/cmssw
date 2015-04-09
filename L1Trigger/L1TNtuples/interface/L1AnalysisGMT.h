#ifndef __L1Analysis_L1AnalysisGMT_H__
#define __L1Analysis_L1AnalysisGMT_H__

//-------------------------------------------------------------------------------
// Created 06/01/2010 - A.C. Le Bihan
// 
//
// Original code : L1TriggerDPG/L1Ntuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "L1AnalysisGMTDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisGMT
  {
  public:
    L1AnalysisGMT();
    ~L1AnalysisGMT();
    
    void Set(const L1MuGMTReadoutCollection* gmtrc, unsigned maxDTBX, unsigned maxCSC, unsigned maxRPC, unsigned maxGMT, bool physVal);	
    L1AnalysisGMTDataFormat * getData() {return &gmt_;}
    void Reset() {gmt_.Reset();}
  
  private :
    L1AnalysisGMTDataFormat gmt_;	   		      
  }; 
} 
#endif


