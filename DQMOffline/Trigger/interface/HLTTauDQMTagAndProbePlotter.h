// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMTagAndProbePlotter_h
#define DQMOffline_Trigger_HLTTauDQMTagAndProbePlotter_h

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

class HLTTauDQMTagAndProbePlotter: private HLTTauDQMPlotter {
public:
  HLTTauDQMTagAndProbePlotter(const std::string& pathNameNum, const std::string& pathNameDen, const HLTConfigProvider& HLTCP,
                       bool doRefAnalysis, const std::string& dqmBaseFolder,
                       const std::string& hltProcess, int nbins,
                       double xmax,
                       std::string& xvariableName);
  ~HLTTauDQMTagAndProbePlotter();

  using HLTTauDQMPlotter::isValid;

  void bookHistograms(DQMStore::IBooker &iBooker);

  void analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const HLTTauDQMOfflineObjects& refCollection);

private:
  const int nbins_;
  const double xmax_;
  const bool doRefAnalysis_;
  std::string xvariable;


  HLTTauDQMPath hltDenominatorPath_;
  HLTTauDQMPath hltNumeratorPath_;

  std::string pathNameDen;
  std::string pathNameNum;

  MonitorElement *h_num;
  MonitorElement *h_den;
};

#endif
