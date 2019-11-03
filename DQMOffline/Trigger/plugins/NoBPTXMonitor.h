#ifndef DQMOFFLINE_TRIGGER_NOBPTXMONITOR_H
#define DQMOFFLINE_TRIGGER_NOBPTXMONITOR_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/interface/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <string>
#include <vector>

class NoBPTXMonitor : public DQMEDAnalyzer, public TriggerDQMBase {

 public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  NoBPTXMonitor(const edm::ParameterSet&);
  ~NoBPTXMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

 private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::CaloJetCollection> jetToken_;
  edm::EDGetTokenT<reco::TrackCollection> muonToken_;

  std::vector<double> jetE_variable_binning_;
  MEbinning jetE_binning_;
  MEbinning jetEta_binning_;
  MEbinning jetPhi_binning_;
  std::vector<double> muonPt_variable_binning_;
  MEbinning muonPt_binning_;
  MEbinning muonEta_binning_;
  MEbinning muonPhi_binning_;
  MEbinning ls_binning_;
  MEbinning bx_binning_;

  ObjME jetENoBPTX_;
  ObjME jetENoBPTX_variableBinning_;
  ObjME jetEVsLS_;
  ObjME jetEVsBX_;
  ObjME jetEtaNoBPTX_;
  ObjME jetEtaVsLS_;
  ObjME jetEtaVsBX_;
  ObjME jetPhiNoBPTX_;
  ObjME jetPhiVsLS_;
  ObjME jetPhiVsBX_;
  ObjME muonPtNoBPTX_;
  ObjME muonPtNoBPTX_variableBinning_;
  ObjME muonPtVsLS_;
  ObjME muonPtVsBX_;
  ObjME muonEtaNoBPTX_;
  ObjME muonEtaVsLS_;
  ObjME muonEtaVsBX_;
  ObjME muonPhiNoBPTX_;
  ObjME muonPhiVsLS_;
  ObjME muonPhiVsBX_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::CaloJet, true> jetSelection_;
  StringCutObjectSelector<reco::Track, true> muonSelection_;

  unsigned int njets_;
  unsigned int nmuons_;
};

#endif  //DQMOFFLINE_TRIGGER_NOBPTXMONITOR_H
