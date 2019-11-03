#ifndef DQMOFFLINE_TRIGGER_DIDISPSTAMUONMONITOR_H
#define DQMOFFLINE_TRIGGER_DIDISPSTAMUONMONITOR_H

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/interface/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class DiDispStaMuonMonitor : public DQMEDAnalyzer, public TriggerDQMBase {

 public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  DiDispStaMuonMonitor(const edm::ParameterSet&);
  ~DiDispStaMuonMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

 private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::TrackCollection> muonToken_;

  std::vector<double> muonPt_variable_binning_;
  MEbinning muonPt_binning_;
  MEbinning muonEta_binning_;
  MEbinning muonPhi_binning_;
  MEbinning muonDxy_binning_;
  MEbinning ls_binning_;

  ObjME muonPtME_;
  ObjME muonPtNoDxyCutME_;
  ObjME muonPtME_variableBinning_;
  ObjME muonPtVsLS_;
  ObjME muonEtaME_;
  ObjME muonPhiME_;
  ObjME muonDxyME_;
  ObjME subMuonPtME_;
  ObjME subMuonPtME_variableBinning_;
  ObjME subMuonEtaME_;
  ObjME subMuonPhiME_;
  ObjME subMuonDxyME_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::Track, true> muonSelectionGeneral_;
  StringCutObjectSelector<reco::Track, true> muonSelectionPt_;
  StringCutObjectSelector<reco::Track, true> muonSelectionDxy_;

  unsigned int nmuons_;
};

#endif
