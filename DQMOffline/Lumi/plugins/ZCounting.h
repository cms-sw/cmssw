#ifndef ZCounting_H
#define ZCounting_H

#include "FWCore/Framework/interface/MakerMacros.h"   // definitions for declaring plug-in modules
#include "FWCore/Framework/interface/Frameworkfwd.h"  // declaration of EDM types
#include "FWCore/Framework/interface/EDAnalyzer.h"    // EDAnalyzer class
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"  // Parameters
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>  // string class
#include <cassert>

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMOffline/Lumi/interface/TriggerDefs.h"
#include "DQMOffline/Lumi/interface/TTrigger.h"
#include "DQMOffline/Lumi/interface/TriggerTools.h"
#include "DQMOffline/Lumi/interface/ElectronIdentifier.h"

class TFile;
class TH1D;
class TTree;
class TClonesArray;
namespace edm {
  class TriggerResults;
  class TriggerNames;
}  // namespace edm
namespace ZCountingTrigger {
  class TTrigger;
}

class ZCounting : public DQMEDAnalyzer {
public:
  ZCounting(const edm::ParameterSet& ps);
  ~ZCounting() override;

  enum MuonIDTypes { NoneID, LooseID, MediumID, TightID, CustomTightID };
  enum MuonIsoTypes { NoneIso, TrackerIso, PFIso };

protected:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;

private:
  //other functions
  void analyzeMuons(edm::Event const& e, edm::EventSetup const& eSetup);
  void analyzeElectrons(edm::Event const& e, edm::EventSetup const& eSetup);
  bool isMuonTrigger(const ZCountingTrigger::TTrigger& triggerMenu, const TriggerBits& hltBits);
  bool isMuonTriggerObj(const ZCountingTrigger::TTrigger& triggerMenu, const TriggerObjects& hltMatchBits);
  bool passMuonID(const reco::Muon& muon, const reco::Vertex& vtx, const MuonIDTypes& idType);
  bool passMuonIso(const reco::Muon& muon, const MuonIsoTypes& isoType, const float isoCut);
  bool isCustomTightMuon(const reco::Muon& muon);

  // Electron-specific functions
  bool isElectronTrigger(ZCountingTrigger::TTrigger triggerMenu, TriggerBits hltBits);
  bool isElectronTriggerObj(ZCountingTrigger::TTrigger triggerMenu, TriggerObjects hltMatchBits);
  bool ele_probe_selection(double pt, double abseta);
  bool ele_tag_selection(double pt, double abseta);

  // initialization from HLT menu; needs to be called on every change in HLT menu
  void initHLT(const edm::TriggerResults&, const edm::TriggerNames&);

  // EDM object collection names
  edm::ParameterSetID fTriggerNamesID;
  edm::InputTag fHLTObjTag;
  edm::InputTag fHLTTag;
  edm::EDGetTokenT<trigger::TriggerEvent> fHLTObjTag_token;
  edm::EDGetTokenT<edm::TriggerResults> fHLTTag_token;
  std::string fPVName;
  edm::EDGetTokenT<reco::VertexCollection> fPVName_token;

  // Muons
  std::string fMuonName;
  edm::EDGetTokenT<reco::MuonCollection> fMuonName_token;
  std::vector<std::string> fMuonHLTNames;
  std::vector<std::string> fMuonHLTObjectNames;

  // Tracks
  std::string fTrackName;
  edm::EDGetTokenT<reco::TrackCollection> fTrackName_token;

  // Electrons
  std::string fElectronName;
  edm::EDGetTokenT<edm::View<reco::GsfElectron>> fGsfElectronName_token;
  std::string fSCName;
  edm::EDGetTokenT<edm::View<reco::SuperCluster>> fSCName_token;

  edm::InputTag fRhoTag;
  edm::EDGetTokenT<double> fRhoToken;

  edm::InputTag fBeamspotTag;
  edm::EDGetTokenT<reco::BeamSpot> fBeamspotToken;

  edm::InputTag fConversionTag;
  edm::EDGetTokenT<reco::ConversionCollection> fConversionToken;

  // bacon fillers
  std::unique_ptr<ZCountingTrigger::TTrigger> fTrigger;

  std::string IDTypestr_;
  std::string IsoTypestr_;
  MuonIDTypes IDType_{NoneID};
  MuonIsoTypes IsoType_{NoneIso};
  double IsoCut_;

  double PtCutL1_;
  double PtCutL2_;
  double EtaCutL1_;
  double EtaCutL2_;

  int MassBin_;
  double MassMin_;
  double MassMax_;

  int LumiBin_;
  double LumiMin_;
  double LumiMax_;

  int PVBin_;
  double PVMin_;
  double PVMax_;

  double VtxNTracksFitCut_;
  double VtxNdofCut_;
  double VtxAbsZCut_;
  double VtxRhoCut_;

  const double MUON_MASS = 0.105658369;
  const double MUON_BOUND = 0.9;

  const float ELECTRON_MASS = 0.000511;

  const float ELE_PT_CUT_TAG;
  const float ELE_PT_CUT_PROBE;
  const float ELE_ETA_CUT_TAG;
  const float ELE_ETA_CUT_PROBE;
  const float ELE_MASS_CUT_LOW;
  const float ELE_MASS_CUT_HIGH;

  const std::string ELE_ID_WP;
  const float ELE_ETA_CRACK_LOW = 1.4442;
  const float ELE_ETA_CRACK_HIGH = 1.56;
  // Electron-specific members
  ElectronIdentifier EleID_;

  // Muon Histograms
  MonitorElement* h_mass_HLT_pass_central;
  MonitorElement* h_mass_HLT_pass_forward;
  MonitorElement* h_mass_HLT_fail_central;
  MonitorElement* h_mass_HLT_fail_forward;

  MonitorElement* h_mass_SIT_pass_central;
  MonitorElement* h_mass_SIT_pass_forward;
  MonitorElement* h_mass_SIT_fail_central;
  MonitorElement* h_mass_SIT_fail_forward;

  MonitorElement* h_mass_Glo_pass_central;
  MonitorElement* h_mass_Glo_pass_forward;
  MonitorElement* h_mass_Glo_fail_central;
  MonitorElement* h_mass_Glo_fail_forward;

  MonitorElement* h_npv;
  MonitorElement* h_npv_yield_Z;
  MonitorElement* h_mass_yield_Z;
  MonitorElement* h_yieldBB_Z;
  MonitorElement* h_yieldEE_Z;

  // Electron Histograms
  MonitorElement* h_ee_mass_id_pass_central;
  MonitorElement* h_ee_mass_id_fail_central;
  MonitorElement* h_ee_mass_id_pass_forward;
  MonitorElement* h_ee_mass_id_fail_forward;

  MonitorElement* h_ee_mass_HLT_pass_central;
  MonitorElement* h_ee_mass_HLT_fail_central;
  MonitorElement* h_ee_mass_HLT_pass_forward;
  MonitorElement* h_ee_mass_HLT_fail_forward;

  MonitorElement* h_ee_yield_Z_ebeb;
  MonitorElement* h_ee_yield_Z_ebee;
  MonitorElement* h_ee_yield_Z_eeee;
};

#endif
