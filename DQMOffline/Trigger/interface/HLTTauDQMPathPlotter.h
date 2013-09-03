// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMPathPlotter_h
#define DQMOffline_Trigger_HLTTauDQMPathPlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPath.h"


namespace edm {
  class Event;
  class EventSetup;
  class TriggerResults;
}

namespace trigger {
  class TriggerEvent;
}

class HLTConfigProvider;

class HLTTauDQMPathPlotter: private HLTTauDQMPlotter {
public:
  HLTTauDQMPathPlotter(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder,
                        const std::string& hltProcess, int ptbins, int etabins, int phibins,
                        double l1MatchDr, double hltMatchDr);
  ~HLTTauDQMPathPlotter();

  using HLTTauDQMPlotter::isValid;

  void beginRun(const HLTConfigProvider& HLTCP);

  void analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const HLTTauDQMOfflineObjects& refCollection);

  const HLTTauDQMPath *getPathObject() const { return &hltPath_; }

  typedef std::tuple<std::string, size_t> FilterIndex;
private:
  const int ptbins_;
  const int etabins_;
  const int phibins_;
  const double l1MatchDr_;
  const double hltMatchDr_;
  const bool doRefAnalysis_;

  HLTTauDQMPath hltPath_;

  MonitorElement *hAcceptedEvents_;
  MonitorElement *hTrigTauEt_;
  MonitorElement *hTrigTauPhi_;
  MonitorElement *hTrigTauEta_;
  MonitorElement *hMass_;
};

#endif
