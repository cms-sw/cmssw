// -*- c++ -*-
#ifndef HLTTauDQMPathSummaryPlotter_h
#define HLTTauDQMPathSummaryPlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

#include<vector>

class HLTTauDQMPath;
namespace edm {
  class TriggerResults;
}
namespace trigger {
  class TriggerEvent;
}

class HLTTauDQMPathSummaryPlotter : public HLTTauDQMPlotter {
public:
    
  HLTTauDQMPathSummaryPlotter(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder, double hltMatchDr);
  ~HLTTauDQMPathSummaryPlotter();
  const std::string name() { return "foo"; }

  void beginRun(const std::vector<const HLTTauDQMPath *>& pathObjects);

  void analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const HLTTauDQMOfflineObjects& refCollection);
private:
  const double hltMatchDr_;
  const bool doRefAnalysis_;

  std::vector<const HLTTauDQMPath *> pathObjects_;

  MonitorElement *all_events;
  MonitorElement *accepted_events;
};
#endif
