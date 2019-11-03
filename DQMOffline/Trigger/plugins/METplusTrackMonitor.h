#ifndef DQMOFFLINE_TRIGGER_METPLUSTRACKMONITOR_H
#define DQMOFFLINE_TRIGGER_METPLUSTRACKMONITOR_H

#include <string>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/interface/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class METplusTrackMonitor : public DQMEDAnalyzer, public TriggerDQMBase {

 public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  METplusTrackMonitor(const edm::ParameterSet &);
  ~METplusTrackMonitor() noexcept(true) override {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

 protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) override;

 private:
  bool getHLTObj(const edm::Handle<trigger::TriggerEvent> &trigSummary, const edm::InputTag &filterTag, trigger::TriggerObject &obj) const;

  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::CaloMETCollection> metToken_;
  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  edm::EDGetTokenT<reco::PFJetCollection> jetToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;

  edm::InputTag hltMetTag_;
  edm::InputTag hltMetCleanTag_;
  edm::InputTag trackLegFilterTag_;

  std::vector<double> met_variable_binning_;
  std::vector<double> muonPt_variable_binning_;

  MEbinning met_binning_;
  MEbinning ls_binning_;
  MEbinning pt_binning_;
  MEbinning eta_binning_;
  MEbinning phi_binning_;

  ObjME metME_variableBinning_;
  ObjME metVsLS_;
  ObjME metPhiME_;
  ObjME deltaphimetj1ME_;
  ObjME metVsHltMet_;
  ObjME metVsHltMetClean_;

  ObjME muonPtME_variableBinning_;
  ObjME muonPtVsLS_;
  ObjME muonEtaME_;
  ObjME deltaphimetmuonME_;
  ObjME muonEtaVsPhi_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::CaloMET, true> metSelection_;
  StringCutObjectSelector<reco::Muon, true> muonSelection_;
  StringCutObjectSelector<reco::PFJet, true> jetSelection_;
  StringCutObjectSelector<reco::Vertex, true> vtxSelection_;

  unsigned nmuons_;
  unsigned njets_;

  double leadJetEtaCut_;

  bool requireLeadMatched_;
  double maxMatchDeltaR_;
};

#endif
