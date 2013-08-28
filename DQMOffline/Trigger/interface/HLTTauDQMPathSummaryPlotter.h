// -*- c++ -*-
#ifndef HLTTauDQMPathSummaryPlotter_h
#define HLTTauDQMPathSummaryPlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

#include<vector>

class HLTTauDQMPath;
namespace edm {
  class TriggerResults;
}

class HLTTauDQMPathSummaryPlotter : public HLTTauDQMPlotter {
public:
    
  HLTTauDQMPathSummaryPlotter(const edm::ParameterSet& pset, const std::string& dqmBaseFolder);
  ~HLTTauDQMPathSummaryPlotter();
  const std::string name() { return "foo"; }

  void beginRun(const std::vector<const HLTTauDQMPath *>& pathObjects);

  void analyze(const edm::TriggerResults& triggerResults);
private:

  std::vector<const HLTTauDQMPath *> pathObjects_;

  MonitorElement *all_events;
  MonitorElement *accepted_events;
};
#endif
