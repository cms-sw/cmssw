// -*- c++ -*-
// Offline DQM For Tau HLT
#ifndef DQMOffline_Trigger_HLTTauDQMOfflineSource_h
#define DQMOffline_Trigger_HLTTauDQMOfflineSource_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//Plotters
#include "DQMOffline/Trigger/interface/HLTTauDQML1Plotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPathPlotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPathSummaryPlotter.h"

//
// class declaration
//

class HLTTauDQMOfflineSource : public edm::EDAnalyzer {
public:
    HLTTauDQMOfflineSource( const edm::ParameterSet& );
    ~HLTTauDQMOfflineSource();

protected:
    void beginJob();
    void beginRun(const edm::Run& r, const edm::EventSetup& c);
    void analyze(const edm::Event& e, const edm::EventSetup& c) ;

private:
    std::string hltProcessName_;
    edm::InputTag triggerResultsSrc_;
    edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
    edm::InputTag triggerEventSrc_;
    edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;
    bool hltMenuChanged_;
    
    HLTConfigProvider HLTCP_;

    //Reference
    bool doRefAnalysis_;
    struct RefObject {
      int objID;
      edm::EDGetTokenT<LVColl> token;
    };
    std::vector<RefObject> refObjects_;

    //DQM Prescaler
    int counterEvt_;      //counter
    const int prescaleEvt_;     //every n events 
    
    // Plotters
    std::vector<HLTTauDQML1Plotter> l1Plotters_;
    std::vector<HLTTauDQMPathPlotter> pathPlotters2_;
    std::vector<HLTTauDQMPathSummaryPlotter> pathSummaryPlotters_;
};

#endif
