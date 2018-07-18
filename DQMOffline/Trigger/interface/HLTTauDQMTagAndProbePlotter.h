// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMTagAndProbePlotter_h
#define DQMOffline_Trigger_HLTTauDQMTagAndProbePlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPath.h"

//#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

namespace edm {
  class Event;
  class EventSetup;
}

namespace trigger {
  class TriggerEvent;
}

class HLTConfigProvider;

class HLTTauDQMTagAndProbePlotter: private HLTTauDQMPlotter {
public:
  HLTTauDQMTagAndProbePlotter(const edm::ParameterSet& iConfig, const std::vector<std::string>& modLabels, const std::string& dqmBaseFolder);
  ~HLTTauDQMTagAndProbePlotter();

  using HLTTauDQMPlotter::isValid;

  void bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &iRun, edm::EventSetup const &iSetup);

  void analyze(edm::Event const& iEvent, const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const HLTTauDQMOfflineObjects& refCollection);


private:
  LV findTrgObject(std::string, const trigger::TriggerEvent&);

  const int nbinsPt_;
  const double ptmin_,ptmax_;
  int nbinsEta_;
  double etamin_,etamax_;
  const int nbinsPhi_;
  const double phimin_,phimax_;
  std::string xvariable;

  std::vector<std::string> numTriggers;
  std::vector<std::string> denTriggers;

  std::vector<std::string> moduleLabels;

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
