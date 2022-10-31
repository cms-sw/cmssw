#ifndef ZCountingElectrons_H
#define ZCountingElectrons_H

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

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMOffline/Lumi/interface/TriggerTools.h"
#include "DQMOffline/Lumi/interface/ElectronIdentifier.h"

class ZCountingElectrons : public DQMEDAnalyzer {
public:
  ZCountingElectrons(const edm::ParameterSet& ps);
  ~ZCountingElectrons() override;

protected:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;

private:
  // Electron-specific functions
  bool ele_probe_selection(double pt, double abseta);
  bool ele_tag_selection(double pt, double abseta);

  // EDM object collection names
  const edm::InputTag triggerResultsInputTag_;
  edm::EDGetTokenT<reco::VertexCollection> fPVName_token;

  // Electrons
  edm::EDGetTokenT<edm::View<reco::GsfElectron>> fGsfElectronName_token;
  edm::EDGetTokenT<edm::View<reco::SuperCluster>> fSCName_token;
  edm::EDGetTokenT<double> fRhoToken;
  edm::EDGetTokenT<reco::BeamSpot> fBeamspotToken;
  edm::EDGetTokenT<reco::ConversionCollection> fConversionToken;

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

  const std::string ELE_ID_WP;

  // trigger objects
  HLTConfigProvider hltConfigProvider_;
  TriggerTools* triggers;

  //constants
  const double DRMAX = 0.1;  // max dR matching between muon and hlt object
  const float ELECTRON_MASS = 0.000511;
  const float ELE_ETA_CRACK_LOW = 1.4442;
  const float ELE_ETA_CRACK_HIGH = 1.56;

  // Electron-specific members
  ElectronIdentifier EleID_;

  // General Histograms
  MonitorElement* h_npv;

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
