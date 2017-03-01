#ifndef __L1Analysis_L1AnalysisL1Menu_H__
#define __L1Analysis_L1AnalysisL1Menu_H__


#include "DataFormats/L1GlobalTrigger/interface/L1GtTriggerMenuLite.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "FWCore/Framework/interface/Event.h"

#include "L1AnalysisL1MenuDataFormat.h"


namespace L1Analysis
{
  class L1AnalysisL1Menu 
  {
  public:
    L1AnalysisL1Menu() {Reset();}
    ~L1AnalysisL1Menu() {}
    void Reset() {data_.Reset();}
    void SetPrescaleFactorIndex(L1GtUtils & l1GtUtils_, const edm::Event& iEvent);
    L1AnalysisL1MenuDataFormat * getData() {return &data_;}

  private :
    L1AnalysisL1MenuDataFormat data_;
  }; 
}
#endif


