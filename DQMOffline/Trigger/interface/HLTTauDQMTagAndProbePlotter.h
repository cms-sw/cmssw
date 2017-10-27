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
  HLTTauDQMTagAndProbePlotter(const edm::ParameterSet& iConfig, std::unique_ptr<GenericTriggerEventFlag> numFlag, std::unique_ptr<GenericTriggerEventFlag> denFlag, const std::string& dqmBaseFolder);
  ~HLTTauDQMTagAndProbePlotter();

  using HLTTauDQMPlotter::isValid;

  void bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &iRun, edm::EventSetup const &iSetup);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup, const HLTTauDQMOfflineObjects& refCollection);


private:
  const int nbinsPt_;
  const double ptmin_,ptmax_;
  int nbinsEta_;
  double etamin_,etamax_;
  const int nbinsPhi_;
  const double phimin_,phimax_;
  std::string xvariable;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  unsigned int nOfflineObjs;

  MonitorElement *h_num_pt;
  MonitorElement *h_den_pt;

  MonitorElement *h_num_eta;
  MonitorElement *h_den_eta;

  MonitorElement *h_num_phi;
  MonitorElement *h_den_phi;

  MonitorElement *h_num_etaphi;
  MonitorElement *h_den_etaphi;
};

#endif
