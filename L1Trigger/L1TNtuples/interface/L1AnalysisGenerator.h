#ifndef __L1Analysis_L1AnalysisGenerator_H__
#define __L1Analysis_L1AnalysisGenerator_H__

//-------------------------------------------------------------------------------
// Created 06/01/2010 - A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------
#include "FWCore/Framework/interface/Event.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisGeneratorDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisGenerator 
  {
  public:
    L1AnalysisGenerator();
    ~L1AnalysisGenerator();
    void Reset() {generator_.Reset();}
    void Set(const edm::Event& e);
    L1AnalysisGeneratorDataFormat * getData(){return &generator_;}
   
  private :
    L1AnalysisGeneratorDataFormat generator_;            
    
  }; 
} 
#endif


