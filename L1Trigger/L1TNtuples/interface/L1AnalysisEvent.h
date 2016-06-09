#ifndef __L1Analysis_L1AnalysisEvent_H__
#define __L1Analysis_L1AnalysisEvent_H__

//-------------------------------------------------------------------------------
// Created 06/01/2010 - A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "PhysicsTools/Utilities/interface/LumiReWeighting.h"

#include "L1AnalysisEventDataFormat.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include <string>

namespace L1Analysis
{
  class L1AnalysisEvent 
  {
  public:
    L1AnalysisEvent(std::string puMCFile, 
		    std::string puMCHist, 
		    std::string puDataFile, 
		    std::string puDataHist,
		    bool useAvgVtx,
		    double maxWeight,
		    edm::ConsumesCollector &&);
    ~L1AnalysisEvent();
    
    //void Print(std::ostream &os = std::cout) const;
    void Set(const edm::Event& e, const edm::EDGetTokenT<edm::TriggerResults>& hlt_);
    void Reset() {event_.Reset();}
    L1AnalysisEventDataFormat * getData() {return &event_;}

    // ---- General L1AnalysisEvent information.
    
  private :
    bool fillHLT_;
    bool doPUWeights_;

    bool   useAvgVtx_;
    double maxAllowedWeight_;

    edm::LumiReWeighting lumiWeights_;

    edm::EDGetTokenT<std::vector<PileupSummaryInfo>> pileupSummaryInfoToken_;
    L1Analysis::L1AnalysisEventDataFormat event_;
    
  }; 
}
#endif


