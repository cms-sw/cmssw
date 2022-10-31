// -*- c++ -*-
// Offline DQM For Tau HLT
#ifndef DQMOffline_Trigger_HLTTauDQMOfflineSource_h
#define DQMOffline_Trigger_HLTTauDQMOfflineSource_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

//Plotters
#include "DQMOffline/Trigger/interface/HistoWrapper.h"
#include "DQMOffline/Trigger/interface/HLTTauDQML1Plotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPathPlotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPathSummaryPlotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMTagAndProbePlotter.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include <regex>

//
// class declaration
//

class HLTTauDQMOfflineSource : public DQMEDAnalyzer {
public:
  HLTTauDQMOfflineSource(const edm::ParameterSet&);
  ~HLTTauDQMOfflineSource() override;

protected:
  void dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) override;
  void bookHistograms(DQMStore::IBooker& iBooker, const edm::Run& r, const edm::EventSetup& c) override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  std::string hltProcessName_;
  edm::InputTag triggerResultsSrc_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::InputTag triggerEventSrc_;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;

  // For path plotters
  const std::string pathRegex_;
  const int nPtBins_, nEtaBins_, nPhiBins_;
  const double ptMax_, highPtMax_, l1MatchDr_, hltMatchDr_;
  const std::string dqmBaseFolder_;

  HistoWrapper* iWrapper;

  HLTConfigProvider HLTCP_;

  //Reference
  bool doRefAnalysis_;
  struct RefObject {
    int objID;
    edm::EDGetTokenT<LVColl> token;
  };
  std::vector<RefObject> refObjects_;
  bool tagAndProbe_;
  std::vector<edm::ParameterSet> tagAndProbePaths;

  //DQM Prescaler
  int counterEvt_;         //counter
  const int prescaleEvt_;  //every n events

  // Plotters
  std::unique_ptr<HLTTauDQML1Plotter> l1Plotter_;
  std::vector<HLTTauDQMPathPlotter> pathPlotters_;
  std::unique_ptr<HLTTauDQMPathSummaryPlotter> pathSummaryPlotter_;
  std::vector<std::unique_ptr<HLTTauDQMTagAndProbePlotter> > tagandprobePlotters_;
};

#endif
