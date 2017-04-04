// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMTagAndProbePlotter_h
#define DQMOffline_Trigger_HLTTauDQMTagAndProbePlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPath.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


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
  HLTTauDQMTagAndProbePlotter(const edm::ParameterSet& iConfig, GenericTriggerEventFlag* numFlag, GenericTriggerEventFlag* denFlag, const std::string& dqmBaseFolder);
  ~HLTTauDQMTagAndProbePlotter();

  using HLTTauDQMPlotter::isValid;

  void bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &iRun, edm::EventSetup const &iSetup);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup, const HLTTauDQMOfflineObjects& refCollection);


private:
  const int nbins_;
  const double xmin_;
  const double xmax_;
  std::string xvariable;

  GenericTriggerEventFlag* num_genTriggerEventFlag_;
  GenericTriggerEventFlag* den_genTriggerEventFlag_;

  MonitorElement *h_num;
  MonitorElement *h_den;
};

#endif
