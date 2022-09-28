// -*- c++ -*-
#ifndef HLTTauDQMPathSummaryPlotter_h
#define HLTTauDQMPathSummaryPlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"
#include "DQMOffline/Trigger/interface/HistoWrapper.h"

#include <vector>

#include "DQMOffline/Trigger/interface/HLTTauDQMPath.h"

//class HLTTauDQMPath;
namespace edm {
  class TriggerResults;
}
namespace trigger {
  class TriggerEvent;
}

class HLTTauDQMPathSummaryPlotter : private HLTTauDQMPlotter {
public:
  HLTTauDQMPathSummaryPlotter(const edm::ParameterSet& pset,
                              bool doRefAnalysis,
                              const std::string& dqmBaseFolder,
                              double hltMatchDr);
  ~HLTTauDQMPathSummaryPlotter();

  using HLTTauDQMPlotter::isValid;

  void setPathObjects(const std::vector<const HLTTauDQMPath*>& pathObjects) { pathObjects_ = pathObjects; }
  void bookHistograms(HistoWrapper& iWrapper, DQMStore::IBooker& iBooker);

  template <class T>
  void analyze(const edm::TriggerResults& triggerResults,
               const T& triggerEvent,  //trigger::TriggerEvent& triggerEvent,
               const HLTTauDQMOfflineObjects& refCollection) {
    if (doRefAnalysis_) {
      std::vector<HLTTauDQMPath::Object> triggerObjs;
      std::vector<HLTTauDQMPath::Object> matchedTriggerObjs;
      HLTTauDQMOfflineObjects matchedOfflineObjs;

      for (size_t i = 0; i < pathObjects_.size(); ++i) {
        const HLTTauDQMPath* path = pathObjects_[i];
        const int lastFilter = path->filtersSize() - 1;

        if (path->goodOfflineEvent(lastFilter, refCollection)) {
          if (all_events)
            all_events->Fill(i + 0.5);
        }
        if (path->fired(triggerResults)) {
          triggerObjs.clear();
          matchedTriggerObjs.clear();
          matchedOfflineObjs.clear();
          path->getFilterObjects(triggerEvent, lastFilter, triggerObjs);
          if (path->offlineMatching(
                  lastFilter, triggerObjs, refCollection, hltMatchDr_, matchedTriggerObjs, matchedOfflineObjs)) {
            if (accepted_events)
              accepted_events->Fill(i + 0.5);
          }
        }
      }
    } else {
      for (size_t i = 0; i < pathObjects_.size(); ++i) {
        const HLTTauDQMPath* path = pathObjects_[i];
        if (all_events)
          all_events->Fill(i + 0.5);
        if (path->fired(triggerResults)) {
          if (accepted_events)
            accepted_events->Fill(i + 0.5);
        }
      }
    }
  }

private:
  const double hltMatchDr_;
  const bool doRefAnalysis_;

  std::vector<const HLTTauDQMPath*> pathObjects_;

  MonitorElement* all_events;
  MonitorElement* accepted_events;
};
#endif
