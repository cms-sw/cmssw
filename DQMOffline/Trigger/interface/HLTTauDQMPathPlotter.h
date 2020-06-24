// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMPathPlotter_h
#define DQMOffline_Trigger_HLTTauDQMPathPlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPath.h"
#include "DQMOffline/Trigger/interface/HistoWrapper.h"

namespace edm {
  class Event;
  class EventSetup;
  class TriggerResults;
}  // namespace edm

namespace trigger {
  class TriggerEvent;
}

class HLTConfigProvider;

class HLTTauDQMPathPlotter : private HLTTauDQMPlotter {
public:
  HLTTauDQMPathPlotter(const std::string &pathName,
                       const HLTConfigProvider &HLTCP,
                       bool doRefAnalysis,
                       const std::string &dqmBaseFolder,
                       const std::string &hltProcess,
                       int ptbins,
                       int etabins,
                       int phibins,
                       double ptmax,
                       double highptmax,
                       double l1MatchDr,
                       double hltMatchDr);
  ~HLTTauDQMPathPlotter();

  using HLTTauDQMPlotter::isValid;

  void bookHistograms(HistoWrapper &iWrapper, DQMStore::IBooker &iBooker);

  void analyze(const edm::TriggerResults &triggerResults,
               const trigger::TriggerEvent &triggerEvent,
               const HLTTauDQMOfflineObjects &refCollection);

  const HLTTauDQMPath *getPathObject() const { return &hltPath_; }

  typedef std::tuple<std::string, size_t> FilterIndex;

private:
  const int ptbins_;
  const int etabins_;
  const int phibins_;
  const double ptmax_;
  const double highptmax_;
  const double l1MatchDr_;
  const double hltMatchDr_;
  const bool doRefAnalysis_;

  HLTTauDQMPath hltPath_;

  MonitorElement *hAcceptedEvents_;
  MonitorElement *hTrigTauEt_;
  MonitorElement *hTrigTauEta_;
  MonitorElement *hTrigTauPhi_;
  MonitorElement *hTrigMuonEt_;
  MonitorElement *hTrigMuonEta_;
  MonitorElement *hTrigMuonPhi_;
  MonitorElement *hTrigElectronEt_;
  MonitorElement *hTrigElectronEta_;
  MonitorElement *hTrigElectronPhi_;
  MonitorElement *hTrigMETEt_;
  MonitorElement *hTrigMETPhi_;
  MonitorElement *hMass_;

  MonitorElement *hL2TrigTauEtEffNum_;
  MonitorElement *hL2TrigTauEtEffDenom_;
  MonitorElement *hL2TrigTauHighEtEffNum_;
  MonitorElement *hL2TrigTauHighEtEffDenom_;
  MonitorElement *hL2TrigTauEtaEffNum_;
  MonitorElement *hL2TrigTauEtaEffDenom_;
  MonitorElement *hL2TrigTauPhiEffNum_;
  MonitorElement *hL2TrigTauPhiEffDenom_;

  MonitorElement *hL3TrigTauEtEffNum_;
  MonitorElement *hL3TrigTauEtEffDenom_;
  MonitorElement *hL3TrigTauHighEtEffNum_;
  MonitorElement *hL3TrigTauHighEtEffDenom_;
  MonitorElement *hL3TrigTauEtaEffNum_;
  MonitorElement *hL3TrigTauEtaEffDenom_;
  MonitorElement *hL3TrigTauPhiEffNum_;
  MonitorElement *hL3TrigTauPhiEffDenom_;
  MonitorElement *hL3TrigTauEtaPhiEffNum_;
  MonitorElement *hL3TrigTauEtaPhiEffDenom_;

  MonitorElement *hL2TrigElectronEtEffNum_;
  MonitorElement *hL2TrigElectronEtEffDenom_;
  MonitorElement *hL2TrigElectronEtaEffNum_;
  MonitorElement *hL2TrigElectronEtaEffDenom_;
  MonitorElement *hL2TrigElectronPhiEffNum_;
  MonitorElement *hL2TrigElectronPhiEffDenom_;

  MonitorElement *hL3TrigElectronEtEffNum_;
  MonitorElement *hL3TrigElectronEtEffDenom_;
  MonitorElement *hL3TrigElectronEtaEffNum_;
  MonitorElement *hL3TrigElectronEtaEffDenom_;
  MonitorElement *hL3TrigElectronPhiEffNum_;
  MonitorElement *hL3TrigElectronPhiEffDenom_;

  MonitorElement *hL2TrigMuonEtEffNum_;
  MonitorElement *hL2TrigMuonEtEffDenom_;
  MonitorElement *hL2TrigMuonEtaEffNum_;
  MonitorElement *hL2TrigMuonEtaEffDenom_;
  MonitorElement *hL2TrigMuonPhiEffNum_;
  MonitorElement *hL2TrigMuonPhiEffDenom_;

  MonitorElement *hL3TrigMuonEtEffNum_;
  MonitorElement *hL3TrigMuonEtEffDenom_;
  MonitorElement *hL3TrigMuonEtaEffNum_;
  MonitorElement *hL3TrigMuonEtaEffDenom_;
  MonitorElement *hL3TrigMuonPhiEffNum_;
  MonitorElement *hL3TrigMuonPhiEffDenom_;

  MonitorElement *hL2TrigMETEtEffNum_;
  MonitorElement *hL2TrigMETEtEffDenom_;
};

#endif
