#ifndef __L1Analysis_L1AnalysisSimulation_H__
#define __L1Analysis_L1AnalysisSimulation_H__

//-------------------------------------------------------------------------------
// Created 06/01/2010 - A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include "FWCore/Framework/interface/Event.h"
#include "L1AnalysisSimulationDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisSimulation 
  {
  public:
    L1AnalysisSimulation();
    ~L1AnalysisSimulation();
    void Reset() {sim_.Reset();}
    void Set(const edm::Event& e);
    L1AnalysisSimulationDataFormat * getData() {return &sim_;}
  private :
    L1AnalysisSimulationDataFormat sim_;
  
  }; 
} 
#endif


