// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMPathPlotter2_h
#define DQMOffline_Trigger_HLTTauDQMPathPlotter2_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

#include<vector>
#include<tuple>

namespace edm {
  class Event;
  class EventSetup;
  class TriggerResults;
}

namespace trigger {
  class TriggerEvent;
}

class HLTConfigProvider;

class HLTTauDQMPathPlotter2: public HLTTauDQMPlotter {
public:
  HLTTauDQMPathPlotter2(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder, const HLTConfigProvider& HLTCP);
  ~HLTTauDQMPathPlotter2();

  void analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const std::map<int, LVColl>& refCollection);
  const std::string name() { return "foo"; }

  typedef std::tuple<std::string, size_t> FilterIndex;

private:
  std::vector<FilterIndex> filterIndices_;
  unsigned int pathIndex_;
  const bool doRefAnalysis_;

  MonitorElement *hAcceptedEvents_;
};

#endif
