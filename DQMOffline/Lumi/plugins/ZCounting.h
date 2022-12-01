#ifndef ZCounting_H
#define ZCounting_H

#include "FWCore/Framework/interface/MakerMacros.h"   // definitions for declaring plug-in modules
#include "FWCore/Framework/interface/Frameworkfwd.h"  // declaration of EDM types
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

#include "DQMOffline/Lumi/interface/TriggerTools.h"

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
  bool passMuonID(const reco::Muon& muon, const reco::Vertex* vtx);
  bool passMuonIso(const reco::Muon& muon);
  bool isCustomTightMuon(const reco::Muon& muon);

  // EDM object collection names
  const edm::InputTag triggerResultsInputTag_;
  edm::EDGetTokenT<reco::VertexCollection> fPVName_token;

  // Muons
  edm::EDGetTokenT<reco::MuonCollection> fMuonName_token;
  std::vector<std::string> fMuonHLTNames;
  std::vector<std::string> fMuonHLTObjectNames;

  // Tracks
  edm::EDGetTokenT<reco::TrackCollection> fTrackName_token;

  // other input
  const double PtCutL1_;
  const double PtCutL2_;
  const double EtaCutL1_;
  const double EtaCutL2_;

  const int MassBin_;
  const double MassMin_;
  const double MassMax_;

  const int LumiBin_;
  const double LumiMin_;
  const double LumiMax_;

  const int PVBin_;
  const double PVMin_;
  const double PVMax_;

  const double VtxNTracksFitCut_;
  const double VtxNdofCut_;
  const double VtxAbsZCut_;
  const double VtxRhoCut_;

  const std::string IDTypestr_;
  const std::string IsoTypestr_;
  const double IsoCut_;

  // muon ID and ISO parameters
  MuonIDTypes IDType_{NoneID};
  MuonIsoTypes IsoType_{NoneIso};

  // trigger objects
  HLTConfigProvider hltConfigProvider_;
  TriggerTools* triggers;

  // constants
  const double DRMAX = 0.1;  // max dR matching between muon and hlt object

  const double MUON_MASS = 0.105658369;
  const double MUON_BOUND = 0.9;

  // General Histograms
  MonitorElement* h_npv;

  // Muon Histograms
  MonitorElement* h_mass_2HLT_BB;
  MonitorElement* h_mass_2HLT_BE;
  MonitorElement* h_mass_2HLT_EE;

  MonitorElement* h_mass_1HLT_BB;
  MonitorElement* h_mass_1HLT_BE;
  MonitorElement* h_mass_1HLT_EE;

  MonitorElement* h_mass_SIT_fail_BB;
  MonitorElement* h_mass_SIT_fail_BE;
  MonitorElement* h_mass_SIT_fail_EE;

  MonitorElement* h_mass_Glo_fail_BB;
  MonitorElement* h_mass_Glo_fail_BE;
  MonitorElement* h_mass_Glo_fail_EE;
};

#endif
