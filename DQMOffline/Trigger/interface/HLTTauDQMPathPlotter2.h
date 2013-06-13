// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMPathPlotter2_h
#define DQMOffline_Trigger_HLTTauDQMPathPlotter2_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

namespace edm {
  class Event;
  class EventSetup;
}

namespace trigger {
  class TriggerEvent;
}

class HLTConfigProvider;

class HLTTauDQMPathPlotter2: public HLTTauDQMPlotter {
public:
  HLTTauDQMPathPlotter2(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder, const HLTConfigProvider& HLTCP);
  ~HLTTauDQMPathPlotter2();

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::map<int, LVColl>& refCollection);
  const std::string name() { return "foo"; }

private:
  const bool doRefAnalysis_;
};

#endif
