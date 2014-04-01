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

class HLTTauDQMPathSummaryPlotter: private HLTTauDQMPlotter {
public:
    
  HLTTauDQMPathSummaryPlotter(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder, double hltMatchDr);
  ~HLTTauDQMPathSummaryPlotter();

  using HLTTauDQMPlotter::isValid;

  void setPathObjects(const std::vector<const HLTTauDQMPath *>& pathObjects) {
    pathObjects_ = pathObjects;
    runValid_ = !pathObjects_.empty();
  }
  void bookHistograms(DQMStore::IBooker &iBooker);

  void analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const HLTTauDQMOfflineObjects& refCollection);
private:
  const double hltMatchDr_;
  const bool doRefAnalysis_;

  std::vector<const HLTTauDQMPath *> pathObjects_;

  MonitorElement *all_events;
  MonitorElement *accepted_events;
};
#endif
