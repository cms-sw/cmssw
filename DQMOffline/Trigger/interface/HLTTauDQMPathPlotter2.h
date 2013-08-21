// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMPathPlotter2_h
#define DQMOffline_Trigger_HLTTauDQMPathPlotter2_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

#include<vector>
#include<tuple>

#include <boost/regex.hpp>

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
  HLTTauDQMPathPlotter2(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder,
                        const std::string& hltProcess, int ptbins, int etabins, int phibins);
  ~HLTTauDQMPathPlotter2();

  void beginRun(const HLTConfigProvider& HLTCP);

  void analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const std::map<int, LVColl>& refCollection);
  const std::string name() { return "foo"; }

  typedef std::tuple<std::string, size_t> FilterIndex;
private:
  const std::string hltProcess_;
  const int ptbins_;
  const int etabins_;
  const int phibins_;
  const bool doRefAnalysis_;

  std::vector<boost::regex> pathRegexs_;
  std::vector<boost::regex> ignoreFilterTypes_;
  std::vector<boost::regex> ignoreFilterNames_;

  std::vector<FilterIndex> filterIndices_;
  std::string pathName_;
  unsigned int pathIndex_;

  MonitorElement *hAcceptedEvents_;
  MonitorElement *hTrigTauEt_;
  MonitorElement *hTrigTauPhi_;
  MonitorElement *hTrigTauEta_;
};

#endif
