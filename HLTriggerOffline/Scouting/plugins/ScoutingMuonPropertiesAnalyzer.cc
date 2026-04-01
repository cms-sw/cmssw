// -*- C++ -*-
//
// Package:    HLTriggerOffline/Scouting
// Class:      ScoutingMuonPropertiesAnalyzer
//
/**\class  HLTriggerOffline/Scouting/plugins/ScoutingMuonPropertiesAnalyzer

 Description: DQMEDAnalyzer module for monitoring scouting muon properties

 Original Author:  Prijith Pradeep, refactored by Copilot
         Created:  Wed, 05 Jun 2024 21:53:24 GMT
*/

// system includes
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include "TLorentzVector.h"

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

typedef math::XYZPoint Point;
typedef math::Error<3>::type Error3;

class ScoutingMuonPropertiesAnalyzer : public DQMEDAnalyzer {
public:
  explicit ScoutingMuonPropertiesAnalyzer(const edm::ParameterSet&);
  ~ScoutingMuonPropertiesAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  template <typename T>
  bool getValidHandle(const edm::Event& iEvent,
                      const edm::EDGetTokenT<T>& token,
                      edm::Handle<T>& handle,
                      const std::string& label);

  // Output Folder
  const std::string outputInternalPath_;
  // do to some histogram duplicates with the ScoutingCollectionMonitor.cc module, we added the option to just fill the unique plots w.r.t the aforementioned module, if this bool is set to false
  const bool fillAllHistograms_;

  // Tokens
  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> muonsNoVtxToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> muonsVtxToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> PVToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> SVNoVtxToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> SVVtxToken_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbESToken_;

  // Triggers
  std::vector<std::string> triggerPathsVector;
  std::map<std::string, int> triggerPathsMap;
  std::map<std::string, MonitorElement*> triggerMEMap_;
  bool triggersMapped_ = false;

  // Scalar event ME
  MonitorElement* h_run_;
  MonitorElement* h_event_;
  MonitorElement* h_lumi_;

  // ScoutingMuonNoVtx MEs
  MonitorElement* h_nScoutingMuonNoVtx_;
  MonitorElement* h_ScoutingMuonNoVtx_pt_;
  MonitorElement* h_ScoutingMuonNoVtx_eta_;
  MonitorElement* h_ScoutingMuonNoVtx_phi_;
  MonitorElement* h_ScoutingMuonNoVtx_phiCorr_;
  MonitorElement* h_ScoutingMuonNoVtx_charge_;
  MonitorElement* h_ScoutingMuonNoVtx_m_;
  MonitorElement* h_ScoutingMuonNoVtx_trkchi2_;
  MonitorElement* h_ScoutingMuonNoVtx_trkndof_;
  MonitorElement* h_ScoutingMuonNoVtx_trkdxy_;
  MonitorElement* h_ScoutingMuonNoVtx_trkdz_;
  MonitorElement* h_ScoutingMuonNoVtx_trkqoverp_;
  MonitorElement* h_ScoutingMuonNoVtx_trklambda_;
  MonitorElement* h_ScoutingMuonNoVtx_trkpt_;
  MonitorElement* h_ScoutingMuonNoVtx_trkphi_;
  MonitorElement* h_ScoutingMuonNoVtx_trketa_;
  MonitorElement* h_ScoutingMuonNoVtx_trkqoverpError_;
  MonitorElement* h_ScoutingMuonNoVtx_trklambdaError_;
  MonitorElement* h_ScoutingMuonNoVtx_trkdxyError_;
  MonitorElement* h_ScoutingMuonNoVtx_trkdzError_;
  MonitorElement* h_ScoutingMuonNoVtx_trkphiError_;
  MonitorElement* h_ScoutingMuonNoVtx_trkdsz_;
  MonitorElement* h_ScoutingMuonNoVtx_trkdszError_;
  MonitorElement* h_ScoutingMuonNoVtx_trkvx_;
  MonitorElement* h_ScoutingMuonNoVtx_trkvy_;
  MonitorElement* h_ScoutingMuonNoVtx_trkvz_;
  MonitorElement* h_ScoutingMuonNoVtx_vtxIndx_;  // as multiplicity per muon

  // ScoutingMuonVtx MEs
  MonitorElement* h_nScoutingMuonVtx_;
  MonitorElement* h_ScoutingMuonVtx_pt_;
  MonitorElement* h_ScoutingMuonVtx_eta_;
  MonitorElement* h_ScoutingMuonVtx_phi_;
  MonitorElement* h_ScoutingMuonVtx_phiCorr_;
  MonitorElement* h_ScoutingMuonVtx_charge_;
  MonitorElement* h_ScoutingMuonVtx_m_;
  MonitorElement* h_ScoutingMuonVtx_trkchi2_;
  MonitorElement* h_ScoutingMuonVtx_trkndof_;
  MonitorElement* h_ScoutingMuonVtx_trkdxy_;
  MonitorElement* h_ScoutingMuonVtx_trkdz_;
  MonitorElement* h_ScoutingMuonVtx_trkqoverp_;
  MonitorElement* h_ScoutingMuonVtx_trklambda_;
  MonitorElement* h_ScoutingMuonVtx_trkpt_;
  MonitorElement* h_ScoutingMuonVtx_trkphi_;
  MonitorElement* h_ScoutingMuonVtx_trketa_;
  MonitorElement* h_ScoutingMuonVtx_trkqoverpError_;
  MonitorElement* h_ScoutingMuonVtx_trklambdaError_;
  MonitorElement* h_ScoutingMuonVtx_trkdxyError_;
  MonitorElement* h_ScoutingMuonVtx_trkdzError_;
  MonitorElement* h_ScoutingMuonVtx_trkphiError_;
  MonitorElement* h_ScoutingMuonVtx_trkdsz_;
  MonitorElement* h_ScoutingMuonVtx_trkdszError_;
  MonitorElement* h_ScoutingMuonVtx_trkvx_;
  MonitorElement* h_ScoutingMuonVtx_trkvy_;
  MonitorElement* h_ScoutingMuonVtx_trkvz_;
  MonitorElement* h_ScoutingMuonVtx_vtxIndx_;  // as multiplicity per muon

  // PV MEs
  MonitorElement* h_nPV_;
  MonitorElement* h_PV_x_;
  MonitorElement* h_PV_y_;
  MonitorElement* h_PV_z_;
  MonitorElement* h_PV_xError_;
  MonitorElement* h_PV_yError_;
  MonitorElement* h_PV_zError_;
  MonitorElement* h_PV_trksize_;
  MonitorElement* h_PV_chi2_;
  MonitorElement* h_PV_ndof_;
  MonitorElement* h_PV_isvalidvtx_;

  // SVNoVtx MEs
  MonitorElement* h_nSVNoVtx_;
  MonitorElement* h_SVNoVtx_x_;
  MonitorElement* h_SVNoVtx_y_;
  MonitorElement* h_SVNoVtx_z_;
  MonitorElement* h_SVNoVtx_xError_;
  MonitorElement* h_SVNoVtx_yError_;
  MonitorElement* h_SVNoVtx_zError_;
  MonitorElement* h_SVNoVtx_trksize_;
  MonitorElement* h_SVNoVtx_chi2_;
  MonitorElement* h_SVNoVtx_ndof_;
  MonitorElement* h_SVNoVtx_isvalidvtx_;
  MonitorElement* h_SVNoVtx_dxy_;
  MonitorElement* h_SVNoVtx_dxySig_;
  MonitorElement* h_SVNoVtx_dlen_;
  MonitorElement* h_SVNoVtx_dlenSig_;
  MonitorElement* h_SVNoVtx_mass_;
  MonitorElement* h_SVNoVtx_mass_JPsi_;
  MonitorElement* h_SVNoVtx_mass_Z_;
  MonitorElement* h_SVNoVtx_nMuon_;

  // SVVtx MEs
  MonitorElement* h_nSVVtx_;
  MonitorElement* h_SVVtx_x_;
  MonitorElement* h_SVVtx_y_;
  MonitorElement* h_SVVtx_z_;
  MonitorElement* h_SVVtx_xError_;
  MonitorElement* h_SVVtx_yError_;
  MonitorElement* h_SVVtx_zError_;
  MonitorElement* h_SVVtx_trksize_;
  MonitorElement* h_SVVtx_chi2_;
  MonitorElement* h_SVVtx_ndof_;
  MonitorElement* h_SVVtx_isvalidvtx_;
  MonitorElement* h_SVVtx_dxy_;
  MonitorElement* h_SVVtx_dxySig_;
  MonitorElement* h_SVVtx_dlen_;
  MonitorElement* h_SVVtx_dlenSig_;
  MonitorElement* h_SVVtx_mass_;
  MonitorElement* h_SVVtx_mass_Z_;
  MonitorElement* h_SVVtx_mass_JPsi_;
  MonitorElement* h_SVVtx_nMuon_;
};

ScoutingMuonPropertiesAnalyzer::ScoutingMuonPropertiesAnalyzer(const edm::ParameterSet& iConfig)
    : outputInternalPath_{iConfig.getParameter<std::string>("OutputInternalPath")},
      fillAllHistograms_{iConfig.getParameter<bool>("fillAllHistograms")},
      triggerResultsToken_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("triggerResults"))),
      muonsNoVtxToken_(consumes<std::vector<Run3ScoutingMuon>>(iConfig.getParameter<edm::InputTag>("muonsNoVtx"))),
      muonsVtxToken_(consumes<std::vector<Run3ScoutingMuon>>(iConfig.getParameter<edm::InputTag>("muonsVtx"))),
      PVToken_(consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("PV"))),
      SVNoVtxToken_(consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("SVNoVtx"))),
      SVVtxToken_(consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("SVVtx"))),
      ttbESToken_(
          esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))) {
  triggerPathsVector = {
      "DST_PFScouting_SingleMuon_v",
      "DST_PFScouting_DoubleMuonNoVtx_v",
      "DST_PFScouting_DoubleMuonVtx_v",
      "DST_PFScouting_DoubleMuonVtxMonitorJPsi_v",
      "DST_PFScouting_DoubleMuonVtxMonitorZ_v",
      "DST_PFScouting_DoubleEG_v",
      "DST_PFScouting_JetHT_v",
      "DST_PFScouting_AXOMedium_v",
      "DST_PFScouting_AXOTight_v",
      "DST_PFScouting_AXOVTight_v",
      "DST_PFScouting_ZeroBias_v",
      "DST_PFScouting_SinglePhotonEB_v",
      "DST_PFScouting_CICADALoose_v",
      "DST_PFScouting_CICADAMedium_v",
      "DST_PFScouting_CICADATight_v",
      "DST_PFScouting_CICADAVTight_v",
      "DST_PFScouting_SingleMuonMonitorJPsi_v",
      "DST_PFScouting_SingleMuonMonitorZ_v",
  };
}

template <typename T>
bool ScoutingMuonPropertiesAnalyzer::getValidHandle(const edm::Event& iEvent,
                                                    const edm::EDGetTokenT<T>& token,
                                                    edm::Handle<T>& handle,
                                                    const std::string& label) {
  iEvent.getByToken(token, handle);
  if (!handle.isValid()) {
    edm::LogWarning("ScoutingMuonPropertiesAnalyzer") << "Invalid handle for " << label << "! Skipping event.";
    return false;
  }
  return true;
}

void ScoutingMuonPropertiesAnalyzer::bookHistograms(DQMStore::IBooker& ibooker,
                                                    edm::Run const&,
                                                    edm::EventSetup const&) {
  ibooker.setCurrentFolder(outputInternalPath_);

  for (const auto& trig : triggerPathsVector)
    triggerMEMap_[trig] = ibooker.book1D(trig, trig + " fired", 2, 0, 2);

  h_run_ = ibooker.book1D("run", "Run number", 1000, 200000, 600000);
  h_run_->setAxisTitle("Run Number", 1);
  h_run_->setAxisTitle("Events", 2);

  h_event_ = ibooker.book1D("event", "Event number", 1000, 0, 100000000000);
  h_event_->setAxisTitle("Event Number", 1);
  h_event_->setAxisTitle("Events", 2);

  h_lumi_ = ibooker.book1D("lumi", "Luminosity block", 1000, 0, 5000);
  h_lumi_->setAxisTitle("Luminosity Block", 1);
  h_lumi_->setAxisTitle("Events", 2);

  if (fillAllHistograms_) {
    // ScoutingMuonNoVtx
    h_nScoutingMuonNoVtx_ = ibooker.book1D("nScoutingMuonNoVtx", "Number of ScoutingMuonNoVtx", 20, 0, 10);
    h_nScoutingMuonNoVtx_->setAxisTitle("Number of Muons", 1);
    h_nScoutingMuonNoVtx_->setAxisTitle("Events", 2);

    h_ScoutingMuonNoVtx_pt_ = ibooker.book1D("ScoutingMuonNoVtx_pt", "MuonNoVtx p_{T}", 100, 0, 100);
    h_ScoutingMuonNoVtx_pt_->setAxisTitle("p_{T} [GeV]", 1);
    h_ScoutingMuonNoVtx_pt_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_eta_ = ibooker.book1D("ScoutingMuonNoVtx_eta", "MuonNoVtx #eta", 80, -3, 3);
    h_ScoutingMuonNoVtx_eta_->setAxisTitle("#eta", 1);
    h_ScoutingMuonNoVtx_eta_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_phi_ = ibooker.book1D("ScoutingMuonNoVtx_phi", "MuonNoVtx #phi", 64, -3.5, 3.5);
    h_ScoutingMuonNoVtx_phi_->setAxisTitle("#phi [rad]", 1);
    h_ScoutingMuonNoVtx_phi_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_charge_ = ibooker.book1D("ScoutingMuonNoVtx_charge", "MuonNoVtx charge", 2, -1, 1);
    h_ScoutingMuonNoVtx_charge_->setAxisTitle("Charge", 1);
    h_ScoutingMuonNoVtx_charge_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkchi2_ = ibooker.book1D("ScoutingMuonNoVtx_trkchi2", "MuonNoVtx track #chi^{2}", 100, 0, 100);
    h_ScoutingMuonNoVtx_trkchi2_->setAxisTitle("Track #chi^{2}", 1);
    h_ScoutingMuonNoVtx_trkchi2_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkndof_ = ibooker.book1D("ScoutingMuonNoVtx_trkndof", "MuonNoVtx track ndof", 40, 0, 50);
    h_ScoutingMuonNoVtx_trkndof_->setAxisTitle("Track ndof", 1);
    h_ScoutingMuonNoVtx_trkndof_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkdxy_ = ibooker.book1D("ScoutingMuonNoVtx_trkdxy", "MuonNoVtx track dxy", 100, -0.7, 0.7);
    h_ScoutingMuonNoVtx_trkdxy_->setAxisTitle("Track dxy [cm]", 1);
    h_ScoutingMuonNoVtx_trkdxy_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkdz_ = ibooker.book1D("ScoutingMuonNoVtx_trkdz", "MuonNoVtx track dz", 100, -40, 40);
    h_ScoutingMuonNoVtx_trkdz_->setAxisTitle("Track dz [cm]", 1);
    h_ScoutingMuonNoVtx_trkdz_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkqoverp_ = ibooker.book1D("ScoutingMuonNoVtx_trkqoverp", "MuonNoVtx track q/p", 100, -1, 1);
    h_ScoutingMuonNoVtx_trkqoverp_->setAxisTitle("Track q/p [1/GeV]", 1);
    h_ScoutingMuonNoVtx_trkqoverp_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trklambda_ =
        ibooker.book1D("ScoutingMuonNoVtx_trklambda", "MuonNoVtx track lambda", 100, -2, 2);
    h_ScoutingMuonNoVtx_trklambda_->setAxisTitle("Track #lambda [rad]", 1);
    h_ScoutingMuonNoVtx_trklambda_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkpt_ = ibooker.book1D("ScoutingMuonNoVtx_trkpt", "MuonNoVtx track pt", 100, 0, 100);
    h_ScoutingMuonNoVtx_trkpt_->setAxisTitle("Track p_{T} [GeV]", 1);
    h_ScoutingMuonNoVtx_trkpt_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkphi_ = ibooker.book1D("ScoutingMuonNoVtx_trkphi", "MuonNoVtx track phi", 64, -3.4, 3.4);
    h_ScoutingMuonNoVtx_trkphi_->setAxisTitle("Track #phi [rad]", 1);
    h_ScoutingMuonNoVtx_trkphi_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trketa_ = ibooker.book1D("ScoutingMuonNoVtx_trketa", "MuonNoVtx track eta", 80, -4, 4);
    h_ScoutingMuonNoVtx_trketa_->setAxisTitle("Track #eta", 1);
    h_ScoutingMuonNoVtx_trketa_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkqoverpError_ =
        ibooker.book1D("ScoutingMuonNoVtx_trkqoverpError", "MuonNoVtx track q/p error", 100, 0, 0.01);
    h_ScoutingMuonNoVtx_trkqoverpError_->setAxisTitle("Track q/p Error", 1);
    h_ScoutingMuonNoVtx_trkqoverpError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trklambdaError_ =
        ibooker.book1D("ScoutingMuonNoVtx_trklambdaError", "MuonNoVtx track lambda error", 100, 0, 0.1);
    h_ScoutingMuonNoVtx_trklambdaError_->setAxisTitle("Track #lambda Error", 1);
    h_ScoutingMuonNoVtx_trklambdaError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkdxyError_ =
        ibooker.book1D("ScoutingMuonNoVtx_trkdxyError", "MuonNoVtx track dxy error", 100, 0, 0.1);
    h_ScoutingMuonNoVtx_trkdxyError_->setAxisTitle("Track dxy Error [cm]", 1);
    h_ScoutingMuonNoVtx_trkdxyError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkdzError_ =
        ibooker.book1D("ScoutingMuonNoVtx_trkdzError", "MuonNoVtx track dz error", 100, 0, 1);
    h_ScoutingMuonNoVtx_trkdzError_->setAxisTitle("Track dz Error [cm]", 1);
    h_ScoutingMuonNoVtx_trkdzError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkphiError_ =
        ibooker.book1D("ScoutingMuonNoVtx_trkphiError", "MuonNoVtx track phi error", 100, 0, 0.1);
    h_ScoutingMuonNoVtx_trkphiError_->setAxisTitle("Track #phi Error [rad]", 1);
    h_ScoutingMuonNoVtx_trkphiError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkdsz_ = ibooker.book1D("ScoutingMuonNoVtx_trkdsz", "MuonNoVtx track dsz", 100, -50, 50);
    h_ScoutingMuonNoVtx_trkdsz_->setAxisTitle("Track dsz [cm]", 1);
    h_ScoutingMuonNoVtx_trkdsz_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkdszError_ =
        ibooker.book1D("ScoutingMuonNoVtx_trkdszError", "MuonNoVtx track dsz error", 100, 0, 1);
    h_ScoutingMuonNoVtx_trkdszError_->setAxisTitle("Track dsz Error [cm]", 1);
    h_ScoutingMuonNoVtx_trkdszError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkvx_ = ibooker.book1D("ScoutingMuonNoVtx_trkvx", "MuonNoVtx track vx", 100, -0.5, 0.5);
    h_ScoutingMuonNoVtx_trkvx_->setAxisTitle("Track vx [cm]", 1);
    h_ScoutingMuonNoVtx_trkvx_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkvy_ = ibooker.book1D("ScoutingMuonNoVtx_trkvy", "MuonNoVtx track vy", 100, -0.5, 0.5);
    h_ScoutingMuonNoVtx_trkvy_->setAxisTitle("Track vy [cm]", 1);
    h_ScoutingMuonNoVtx_trkvy_->setAxisTitle("Muons", 2);

    h_ScoutingMuonNoVtx_trkvz_ = ibooker.book1D("ScoutingMuonNoVtx_trkvz", "MuonNoVtx track vz", 100, -50, 50);
    h_ScoutingMuonNoVtx_trkvz_->setAxisTitle("Track vz [cm]", 1);
    h_ScoutingMuonNoVtx_trkvz_->setAxisTitle("Muons", 2);

    // ScoutingMuonVtx
    h_nScoutingMuonVtx_ = ibooker.book1D("nScoutingMuonVtx", "Number of ScoutingMuonVtx", 20, 0, 20);
    h_nScoutingMuonVtx_->setAxisTitle("Number of Muons", 1);
    h_nScoutingMuonVtx_->setAxisTitle("Events", 2);

    h_ScoutingMuonVtx_pt_ = ibooker.book1D("ScoutingMuonVtx_pt", "MuonVtx p_{T}", 100, 0, 100);
    h_ScoutingMuonVtx_pt_->setAxisTitle("p_{T} [GeV]", 1);
    h_ScoutingMuonVtx_pt_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_eta_ = ibooker.book1D("ScoutingMuonVtx_eta", "MuonVtx #eta", 80, -4, 4);
    h_ScoutingMuonVtx_eta_->setAxisTitle("#eta", 1);
    h_ScoutingMuonVtx_eta_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_phi_ = ibooker.book1D("ScoutingMuonVtx_phi", "MuonVtx #phi", 64, -3.4, 3.4);
    h_ScoutingMuonVtx_phi_->setAxisTitle("#phi [rad]", 1);
    h_ScoutingMuonVtx_phi_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_charge_ = ibooker.book1D("ScoutingMuonVtx_charge", "MuonVtx charge", 2, -1, 1);
    h_ScoutingMuonVtx_charge_->setAxisTitle("Charge", 1);
    h_ScoutingMuonVtx_charge_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkchi2_ = ibooker.book1D("ScoutingMuonVtx_trkchi2", "MuonVtx track #chi^{2}", 100, 0, 100);
    h_ScoutingMuonVtx_trkchi2_->setAxisTitle("Track #chi^{2}", 1);
    h_ScoutingMuonVtx_trkchi2_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkndof_ = ibooker.book1D("ScoutingMuonVtx_trkndof", "MuonVtx track ndof", 40, 0, 60);
    h_ScoutingMuonVtx_trkndof_->setAxisTitle("Track ndof", 1);
    h_ScoutingMuonVtx_trkndof_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkdxy_ = ibooker.book1D("ScoutingMuonVtx_trkdxy", "MuonVtx track dxy", 100, -0.5, 0.5);
    h_ScoutingMuonVtx_trkdxy_->setAxisTitle("Track dxy [cm]", 1);
    h_ScoutingMuonVtx_trkdxy_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkdz_ = ibooker.book1D("ScoutingMuonVtx_trkdz", "MuonVtx track dz", 100, -20, 20);
    h_ScoutingMuonVtx_trkdz_->setAxisTitle("Track dz [cm]", 1);
    h_ScoutingMuonVtx_trkdz_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkqoverp_ = ibooker.book1D("ScoutingMuonVtx_trkqoverp", "MuonVtx track q/p", 100, -0.4, 0.4);
    h_ScoutingMuonVtx_trkqoverp_->setAxisTitle("Track q/p [1/GeV]", 1);
    h_ScoutingMuonVtx_trkqoverp_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trklambda_ = ibooker.book1D("ScoutingMuonVtx_trklambda", "MuonVtx track lambda", 100, -2, 2);
    h_ScoutingMuonVtx_trklambda_->setAxisTitle("Track #lambda [rad]", 1);
    h_ScoutingMuonVtx_trklambda_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkpt_ = ibooker.book1D("ScoutingMuonVtx_trkpt", "MuonVtx track pt", 100, 0, 100);
    h_ScoutingMuonVtx_trkpt_->setAxisTitle("Track p_{T} [GeV]", 1);
    h_ScoutingMuonVtx_trkpt_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkphi_ = ibooker.book1D("ScoutingMuonVtx_trkphi", "MuonVtx track phi", 64, -3.4, 3.4);
    h_ScoutingMuonVtx_trkphi_->setAxisTitle("Track #phi [rad]", 1);
    h_ScoutingMuonVtx_trkphi_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trketa_ = ibooker.book1D("ScoutingMuonVtx_trketa", "MuonVtx track eta", 80, -4, 4);
    h_ScoutingMuonVtx_trketa_->setAxisTitle("Track #eta", 1);
    h_ScoutingMuonVtx_trketa_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkqoverpError_ =
        ibooker.book1D("ScoutingMuonVtx_trkqoverpError", "MuonVtx track q/p error", 100, 0, 0.01);
    h_ScoutingMuonVtx_trkqoverpError_->setAxisTitle("Track q/p Error", 1);
    h_ScoutingMuonVtx_trkqoverpError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trklambdaError_ =
        ibooker.book1D("ScoutingMuonVtx_trklambdaError", "MuonVtx track lambda error", 100, 0, 0.1);
    h_ScoutingMuonVtx_trklambdaError_->setAxisTitle("Track #lambda Error", 1);
    h_ScoutingMuonVtx_trklambdaError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkdxyError_ =
        ibooker.book1D("ScoutingMuonVtx_trkdxyError", "MuonVtx track dxy error", 100, 0, 0.1);
    h_ScoutingMuonVtx_trkdxyError_->setAxisTitle("Track dxy Error [cm]", 1);
    h_ScoutingMuonVtx_trkdxyError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkdzError_ = ibooker.book1D("ScoutingMuonVtx_trkdzError", "MuonVtx track dz error", 100, 0, 1);
    h_ScoutingMuonVtx_trkdzError_->setAxisTitle("Track dz Error [cm]", 1);
    h_ScoutingMuonVtx_trkdzError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkphiError_ =
        ibooker.book1D("ScoutingMuonVtx_trkphiError", "MuonVtx track phi error", 100, 0, 0.1);
    h_ScoutingMuonVtx_trkphiError_->setAxisTitle("Track #phi Error [rad]", 1);
    h_ScoutingMuonVtx_trkphiError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkdsz_ = ibooker.book1D("ScoutingMuonVtx_trkdsz", "MuonVtx track dsz", 100, -20, 20);
    h_ScoutingMuonVtx_trkdsz_->setAxisTitle("Track dsz [cm]", 1);
    h_ScoutingMuonVtx_trkdsz_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkdszError_ =
        ibooker.book1D("ScoutingMuonVtx_trkdszError", "MuonVtx track dsz error", 100, 0, 1);
    h_ScoutingMuonVtx_trkdszError_->setAxisTitle("Track dsz Error [cm]", 1);
    h_ScoutingMuonVtx_trkdszError_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkvx_ = ibooker.book1D("ScoutingMuonVtx_trkvx", "MuonVtx track vx", 100, -0.5, 0.5);
    h_ScoutingMuonVtx_trkvx_->setAxisTitle("Track vx [cm]", 1);
    h_ScoutingMuonVtx_trkvx_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkvy_ = ibooker.book1D("ScoutingMuonVtx_trkvy", "MuonVtx track vy", 100, -0.5, 0.5);
    h_ScoutingMuonVtx_trkvy_->setAxisTitle("Track vy [cm]", 1);
    h_ScoutingMuonVtx_trkvy_->setAxisTitle("Muons", 2);

    h_ScoutingMuonVtx_trkvz_ = ibooker.book1D("ScoutingMuonVtx_trkvz", "MuonVtx track vz", 100, -20, 20);
    h_ScoutingMuonVtx_trkvz_->setAxisTitle("Track vz [cm]", 1);
    h_ScoutingMuonVtx_trkvz_->setAxisTitle("Muons", 2);

    // PV
    h_nPV_ = ibooker.book1D("nPV", "Number of PVs", 10, 0, 70);
    h_nPV_->setAxisTitle("Number of PVs", 1);
    h_nPV_->setAxisTitle("Events", 2);

    h_PV_x_ = ibooker.book1D("PV_x", "PV x", 100, -0.5, 0.5);
    h_PV_x_->setAxisTitle("x [cm]", 1);
    h_PV_x_->setAxisTitle("Vertices", 2);

    h_PV_y_ = ibooker.book1D("PV_y", "PV y", 100, -0.5, 0.5);
    h_PV_y_->setAxisTitle("y [cm]", 1);
    h_PV_y_->setAxisTitle("Vertices", 2);

    h_PV_z_ = ibooker.book1D("PV_z", "PV z", 100, -20, 20);
    h_PV_z_->setAxisTitle("z [cm]", 1);
    h_PV_z_->setAxisTitle("Vertices", 2);

    h_PV_xError_ = ibooker.book1D("PV_xError", "PV x error", 100, 0, 0.01);
    h_PV_xError_->setAxisTitle("x Error [cm]", 1);
    h_PV_xError_->setAxisTitle("Vertices", 2);

    h_PV_yError_ = ibooker.book1D("PV_yError", "PV y error", 100, 0, 0.01);
    h_PV_yError_->setAxisTitle("y Error [cm]", 1);
    h_PV_yError_->setAxisTitle("Vertices", 2);

    h_PV_zError_ = ibooker.book1D("PV_zError", "PV z error", 100, 0, 0.1);
    h_PV_zError_->setAxisTitle("z Error [cm]", 1);
    h_PV_zError_->setAxisTitle("Vertices", 2);

    h_PV_trksize_ = ibooker.book1D("PV_trksize", "PV tracks size", 40, 0, 200);
    h_PV_trksize_->setAxisTitle("PV Track Size", 1);
    h_PV_trksize_->setAxisTitle("Vertices", 2);

    h_PV_chi2_ = ibooker.book1D("PV_chi2", "PV #chi^2", 100, 0, 300);
    h_PV_chi2_->setAxisTitle("PV #chi^{2}", 1);
    h_PV_chi2_->setAxisTitle("Vertices", 2);

    h_PV_ndof_ = ibooker.book1D("PV_ndof", "PV ndof", 40, 0, 100);
    h_PV_ndof_->setAxisTitle("PV ndof", 1);
    h_PV_ndof_->setAxisTitle("Vertices", 2);

    h_PV_isvalidvtx_ = ibooker.book1D("PV_isvalidvtx", "PV is valid vtx", 2, 0, 2);
    h_PV_isvalidvtx_->setAxisTitle("Is Valid Vertex", 1);
    h_PV_isvalidvtx_->setAxisTitle("Vertices", 2);

    // SVNoVtx
    h_nSVNoVtx_ = ibooker.book1D("nSVNoVtx", "Number of SVNoVtx", 20, 0, 20);
    h_nSVNoVtx_->setAxisTitle("Number of SVNoVtx", 1);
    h_nSVNoVtx_->setAxisTitle("Events", 2);

    h_SVNoVtx_x_ = ibooker.book1D("SVNoVtx_x", "SVNoVtx x", 100, -0.5, 0.5);
    h_SVNoVtx_x_->setAxisTitle("x [cm]", 1);
    h_SVNoVtx_x_->setAxisTitle("Vertices", 2);

    h_SVNoVtx_y_ = ibooker.book1D("SVNoVtx_y", "SVNoVtx y", 100, -0.5, 0.5);
    h_SVNoVtx_y_->setAxisTitle("y [cm]", 1);
    h_SVNoVtx_y_->setAxisTitle("Vertices", 2);

    h_SVNoVtx_z_ = ibooker.book1D("SVNoVtx_z", "SVNoVtx z", 100, -20, 20);
    h_SVNoVtx_z_->setAxisTitle("z [cm]", 1);
    h_SVNoVtx_z_->setAxisTitle("Vertices", 2);

    h_SVNoVtx_xError_ = ibooker.book1D("SVNoVtx_xError", "SVNoVtx x error", 100, 0, 0.01);
    h_SVNoVtx_xError_->setAxisTitle("x Error [cm]", 1);
    h_SVNoVtx_xError_->setAxisTitle("Vertices", 2);

    h_SVNoVtx_yError_ = ibooker.book1D("SVNoVtx_yError", "SVNoVtx y error", 100, 0, 0.01);
    h_SVNoVtx_yError_->setAxisTitle("y Error [cm]", 1);
    h_SVNoVtx_yError_->setAxisTitle("Vertices", 2);

    h_SVNoVtx_zError_ = ibooker.book1D("SVNoVtx_zError", "SVNoVtx z error", 100, 0, 0.01);
    h_SVNoVtx_zError_->setAxisTitle("z Error [cm]", 1);
    h_SVNoVtx_zError_->setAxisTitle("Vertices", 2);

    h_SVNoVtx_trksize_ = ibooker.book1D("SVNoVtx_trksize", "SVNoVtx tracks size", 40, 0, 40);
    h_SVNoVtx_trksize_->setAxisTitle("SVNoVtx Track Size", 1);
    h_SVNoVtx_trksize_->setAxisTitle("Vertices", 2);

    h_SVNoVtx_chi2_ = ibooker.book1D("SVNoVtx_chi2", "SVNoVtx #chi^2", 100, 0, 50);
    h_SVNoVtx_chi2_->setAxisTitle("SVNoVtx #chi^{2}", 1);
    h_SVNoVtx_chi2_->setAxisTitle("Vertices", 2);

    h_SVNoVtx_ndof_ = ibooker.book1D("SVNoVtx_ndof", "SVNoVtx ndof", 40, 0, 40);
    h_SVNoVtx_ndof_->setAxisTitle("SVNoVtx ndof", 1);
    h_SVNoVtx_ndof_->setAxisTitle("Vertices", 2);

    h_SVNoVtx_isvalidvtx_ = ibooker.book1D("SVNoVtx_isvalidvtx", "SVNoVtx is valid vtx", 2, 0, 2);
    h_SVNoVtx_isvalidvtx_->setAxisTitle("Is Valid Vertex", 1);
    h_SVNoVtx_isvalidvtx_->setAxisTitle("Vertices", 2);

    // SVVtx
    h_nSVVtx_ = ibooker.book1D("nSVVtx", "Number of SVVtx", 20, 0, 20);
    h_nSVVtx_->setAxisTitle("Number of SVVtx", 1);
    h_nSVVtx_->setAxisTitle("Events", 2);

    h_SVVtx_x_ = ibooker.book1D("SVVtx_x", "SVVtx x", 100, -0.5, 0.5);
    h_SVVtx_x_->setAxisTitle("x [cm]", 1);
    h_SVVtx_x_->setAxisTitle("Vertices", 2);

    h_SVVtx_y_ = ibooker.book1D("SVVtx_y", "SVVtx y", 100, -0.5, 0.5);
    h_SVVtx_y_->setAxisTitle("y [cm]", 1);
    h_SVVtx_y_->setAxisTitle("Vertices", 2);

    h_SVVtx_z_ = ibooker.book1D("SVVtx_z", "SVVtx z", 100, -20, 20);
    h_SVVtx_z_->setAxisTitle("z [cm]", 1);
    h_SVVtx_z_->setAxisTitle("Vertices", 2);

    h_SVVtx_xError_ = ibooker.book1D("SVVtx_xError", "SVVtx x error", 100, 0, 0.01);
    h_SVVtx_xError_->setAxisTitle("x Error [cm]", 1);
    h_SVVtx_xError_->setAxisTitle("Vertices", 2);

    h_SVVtx_yError_ = ibooker.book1D("SVVtx_yError", "SVVtx y error", 100, 0, 0.01);
    h_SVVtx_yError_->setAxisTitle("y Error [cm]", 1);
    h_SVVtx_yError_->setAxisTitle("Vertices", 2);

    h_SVVtx_zError_ = ibooker.book1D("SVVtx_zError", "SVVtx z error", 100, 0, 0.01);
    h_SVVtx_zError_->setAxisTitle("z Error [cm]", 1);
    h_SVVtx_zError_->setAxisTitle("Vertices", 2);

    h_SVVtx_trksize_ = ibooker.book1D("SVVtx_trksize", "SVVtx tracks size", 40, 0, 40);
    h_SVVtx_trksize_->setAxisTitle("SVVtx Track Size", 1);
    h_SVVtx_trksize_->setAxisTitle("Vertices", 2);

    h_SVVtx_chi2_ = ibooker.book1D("SVVtx_chi2", "SVVtx #chi^2", 100, 0, 20);
    h_SVVtx_chi2_->setAxisTitle("SVVtx #chi^{2}", 1);
    h_SVVtx_chi2_->setAxisTitle("Vertices", 2);

    h_SVVtx_ndof_ = ibooker.book1D("SVVtx_ndof", "SVVtx ndof", 40, 0, 40);
    h_SVVtx_ndof_->setAxisTitle("SVVtx ndof", 1);
    h_SVVtx_ndof_->setAxisTitle("Vertices", 2);

    h_SVVtx_isvalidvtx_ = ibooker.book1D("SVVtx_isvalidvtx", "SVVtx is valid vtx", 2, 0, 2);
    h_SVVtx_isvalidvtx_->setAxisTitle("Is Valid Vertex", 1);
    h_SVVtx_isvalidvtx_->setAxisTitle("Vertices", 2);
  }

  // ScoutingMuonNoVtx

  h_ScoutingMuonNoVtx_phiCorr_ =
      ibooker.book1D("ScoutingMuonNoVtx_phiCorr", "MuonNoVtx #phi extrapolated", 64, -3.5, 3.5);
  h_ScoutingMuonNoVtx_phiCorr_->setAxisTitle("#phi extrapolated [rad]", 1);
  h_ScoutingMuonNoVtx_phiCorr_->setAxisTitle("Muons", 2);

  h_ScoutingMuonNoVtx_m_ = ibooker.book1D("ScoutingMuonNoVtx_m", "MuonNoVtx mass", 50, 0, 10);
  h_ScoutingMuonNoVtx_m_->setAxisTitle("Mass [GeV]", 1);
  h_ScoutingMuonNoVtx_m_->setAxisTitle("Muons", 2);

  h_ScoutingMuonNoVtx_vtxIndx_ =
      ibooker.book1D("ScoutingMuonNoVtx_vtxIndx", "MuonNoVtx SV multiplicity per muon", 10, 0, 5);
  h_ScoutingMuonNoVtx_vtxIndx_->setAxisTitle("SV Multiplicity per Muon", 1);
  h_ScoutingMuonNoVtx_vtxIndx_->setAxisTitle("Muons", 2);

  // ScoutingMuonVtx

  h_ScoutingMuonVtx_vtxIndx_ = ibooker.book1D("ScoutingMuonVtx_vtxIndx", "MuonVtx SV multiplicity per muon", 10, 0, 10);
  h_ScoutingMuonVtx_vtxIndx_->setAxisTitle("SV Multiplicity per Muon", 1);
  h_ScoutingMuonVtx_vtxIndx_->setAxisTitle("Muons", 2);

  h_ScoutingMuonVtx_phiCorr_ = ibooker.book1D("ScoutingMuonVtx_phiCorr", "MuonVtx #phi extrapolated", 64, -3.4, 3.4);
  h_ScoutingMuonVtx_phiCorr_->setAxisTitle("#phi extrapolated [rad]", 1);
  h_ScoutingMuonVtx_phiCorr_->setAxisTitle("Muons", 2);

  h_ScoutingMuonVtx_m_ = ibooker.book1D("ScoutingMuonVtx_m", "MuonVtx mass", 50, 0, 10);
  h_ScoutingMuonVtx_m_->setAxisTitle("Mass [GeV]", 1);
  h_ScoutingMuonVtx_m_->setAxisTitle("Muons", 2);

  // SVNoVtx

  h_SVNoVtx_dxy_ = ibooker.book1D("SVNoVtx_dxy", "SVNoVtx dxy", 100, 0, 0.5);
  h_SVNoVtx_dxy_->setAxisTitle("dxy [cm]", 1);
  h_SVNoVtx_dxy_->setAxisTitle("Vertices", 2);

  h_SVNoVtx_dxySig_ = ibooker.book1D("SVNoVtx_dxySig", "SVNoVtx dxy significance", 100, 0, 10);
  h_SVNoVtx_dxySig_->setAxisTitle("dxy Significance", 1);
  h_SVNoVtx_dxySig_->setAxisTitle("Vertices", 2);

  h_SVNoVtx_dlen_ = ibooker.book1D("SVNoVtx_dlen", "SVNoVtx dlen", 100, 0, 20);
  h_SVNoVtx_dlen_->setAxisTitle("Decay Length [cm]", 1);
  h_SVNoVtx_dlen_->setAxisTitle("Vertices", 2);

  h_SVNoVtx_dlenSig_ = ibooker.book1D("SVNoVtx_dlenSig", "SVNoVtx dlen significance", 100, 0, 50);
  h_SVNoVtx_dlenSig_->setAxisTitle("Decay Length Significance", 1);
  h_SVNoVtx_dlenSig_->setAxisTitle("Vertices", 2);

  h_SVNoVtx_mass_ = ibooker.book1D("SVNoVtx_mass", "SVNoVtx mass", 50, 0, 100);
  h_SVNoVtx_mass_->setAxisTitle("Mass [GeV]", 1);
  h_SVNoVtx_mass_->setAxisTitle("Vertices", 2);

  h_SVNoVtx_mass_JPsi_ = ibooker.book1D("SVNoVtx_mass_JPsi", "SVNoVtx mass J/Psi", 50, 0, 10);
  h_SVNoVtx_mass_JPsi_->setAxisTitle("Mass [GeV]", 1);
  h_SVNoVtx_mass_JPsi_->setAxisTitle("Vertices", 2);

  h_SVNoVtx_mass_Z_ = ibooker.book1D("SVNoVtx_mass_Z", "SVNoVtx mass Z", 50, 80, 100);
  h_SVNoVtx_mass_Z_->setAxisTitle("Mass [GeV]", 1);
  h_SVNoVtx_mass_Z_->setAxisTitle("Vertices", 2);

  h_SVNoVtx_nMuon_ = ibooker.book1D("SVNoVtx_nMuon", "SVNoVtx nMuon", 10, 0, 10);
  h_SVNoVtx_nMuon_->setAxisTitle("Number of Muons", 1);
  h_SVNoVtx_nMuon_->setAxisTitle("Vertices", 2);

  // SVVtx

  h_SVVtx_dxy_ = ibooker.book1D("SVVtx_dxy", "SVVtx dxy", 100, 0, 0.5);
  h_SVVtx_dxy_->setAxisTitle("dxy [cm]", 1);
  h_SVVtx_dxy_->setAxisTitle("Vertices", 2);

  h_SVVtx_dxySig_ = ibooker.book1D("SVVtx_dxySig", "SVVtx dxy significance", 100, 0, 10);
  h_SVVtx_dxySig_->setAxisTitle("dxy Significance", 1);
  h_SVVtx_dxySig_->setAxisTitle("Vertices", 2);

  h_SVVtx_dlen_ = ibooker.book1D("SVVtx_dlen", "SVVtx dlen", 100, 0, 20);
  h_SVVtx_dlen_->setAxisTitle("Decay Length [cm]", 1);
  h_SVVtx_dlen_->setAxisTitle("Vertices", 2);

  h_SVVtx_dlenSig_ = ibooker.book1D("SVVtx_dlenSig", "SVVtx dlen significance", 100, 0, 10);
  h_SVVtx_dlenSig_->setAxisTitle("Decay Length Significance", 1);
  h_SVVtx_dlenSig_->setAxisTitle("Vertices", 2);

  h_SVVtx_mass_ = ibooker.book1D("SVVtx_mass", "SVVtx mass", 50, 0, 100);
  h_SVVtx_mass_->setAxisTitle("Mass [GeV]", 1);
  h_SVVtx_mass_->setAxisTitle("Vertices", 2);

  h_SVVtx_mass_JPsi_ = ibooker.book1D("SVVtx_mass_JPsi", "SVVtx mass J/Psi", 50, 0, 10);
  h_SVVtx_mass_JPsi_->setAxisTitle("Mass [GeV]", 1);
  h_SVVtx_mass_JPsi_->setAxisTitle("Vertices", 2);

  h_SVVtx_mass_Z_ = ibooker.book1D("SVVtx_mass_Z", "SVVtx mass Z", 50, 80, 100);
  h_SVVtx_mass_Z_->setAxisTitle("Mass [GeV]", 1);
  h_SVVtx_mass_Z_->setAxisTitle("Vertices", 2);

  h_SVVtx_nMuon_ = ibooker.book1D("SVVtx_nMuon", "SVVtx nMuon", 10, 0, 10);
  h_SVVtx_nMuon_->setAxisTitle("Number of Muons", 1);
  h_SVVtx_nMuon_->setAxisTitle("Vertices", 2);

  triggersMapped_ = false;
}

void ScoutingMuonPropertiesAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<edm::TriggerResults> triggerResults;
  edm::Handle<std::vector<Run3ScoutingMuon>> muonsNoVtx;
  edm::Handle<std::vector<Run3ScoutingMuon>> muonsVtx;
  edm::Handle<std::vector<Run3ScoutingVertex>> SVNoVtx;
  edm::Handle<std::vector<Run3ScoutingVertex>> SVVtx;

  if (!getValidHandle(iEvent, triggerResultsToken_, triggerResults, "TriggerResults") ||
      !getValidHandle(iEvent, muonsNoVtxToken_, muonsNoVtx, "muonsNoVtx") ||
      !getValidHandle(iEvent, muonsVtxToken_, muonsVtx, "muonsVtx") ||
      !getValidHandle(iEvent, SVNoVtxToken_, SVNoVtx, "SVNoVtx") ||
      !getValidHandle(iEvent, SVVtxToken_, SVVtx, "SVVtx")) {
    return;
  }

  edm::Handle<std::vector<Run3ScoutingVertex>> PV;
  iEvent.getByToken(PVToken_, PV);

  const TransientTrackBuilder* theB = &iSetup.getData(ttbESToken_);

  h_run_->Fill(iEvent.eventAuxiliary().run());
  h_event_->Fill(iEvent.eventAuxiliary().event());
  h_lumi_->Fill(iEvent.eventAuxiliary().luminosityBlock());

  // Trigger logic (same as before)
  if (!triggersMapped_ && triggerResults.isValid()) {
    const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
    for (size_t i = 0; i < triggerPathsVector.size(); ++i) {
      triggerPathsMap[triggerPathsVector[i]] = -1;
      for (size_t j = 0; j < triggerNames.size(); ++j) {
        if (triggerNames.triggerName(j).find(triggerPathsVector[i]) != std::string::npos) {
          triggerPathsMap[triggerPathsVector[i]] = j;
        }
      }
    }
    triggersMapped_ = true;
  }
  if (triggerResults.isValid()) {
    for (const auto& trig : triggerPathsVector) {
      int idx = triggerPathsMap[trig];
      if (idx >= 0) {
        bool fired = triggerResults->accept(idx);
        triggerMEMap_[trig]->Fill(fired ? 1 : 0);
      }
    }
  }

  // Prepare PV0 for SV calculations, if available
  reco::Vertex PV0;
  bool pvAvailable = false;
  if (PV.isValid() && !PV->empty()) {
    const auto& PV0Ptr = PV->front();
    Point PV0_pos(PV0Ptr.x(), PV0Ptr.y(), PV0Ptr.z());
    Error3 PV0_err;
    PV0_err(0, 0) = std::pow(PV0Ptr.xError(), 2);
    PV0_err(1, 1) = std::pow(PV0Ptr.yError(), 2);
    PV0_err(2, 2) = std::pow(PV0Ptr.zError(), 2);
    PV0 = reco::Vertex(PV0_pos, PV0_err, PV0Ptr.chi2(), PV0Ptr.ndof(), PV0Ptr.tracksSize());
    pvAvailable = true;
    if (fillAllHistograms_) {
      h_PV_x_->Fill(PV0Ptr.x());
      h_PV_y_->Fill(PV0Ptr.y());
      h_PV_z_->Fill(PV0Ptr.z());
      h_PV_xError_->Fill(PV0Ptr.xError());
      h_PV_yError_->Fill(PV0Ptr.yError());
      h_PV_zError_->Fill(PV0Ptr.zError());
      h_PV_trksize_->Fill(PV0Ptr.tracksSize());
      h_PV_chi2_->Fill(PV0Ptr.chi2());
      h_PV_ndof_->Fill(PV0Ptr.ndof());
      h_PV_isvalidvtx_->Fill(PV0Ptr.isValidVtx());
      h_nPV_->Fill(PV->size());
    }
  }

  // Fill ScoutingMuonNoVtx block, including phiCorr logic
  std::vector<float> muonNoVtx_phiCorr;
  std::vector<std::vector<int>> muonNoVtx_vtxIndx;
  if (muonsNoVtx.isValid()) {
    if (fillAllHistograms_) {
      h_nScoutingMuonNoVtx_->Fill(muonsNoVtx->size());
    }
    for (size_t i = 0; i < muonsNoVtx->size(); ++i) {
      const auto& mu = muonsNoVtx->at(i);
      if (fillAllHistograms_) {
        h_ScoutingMuonNoVtx_pt_->Fill(mu.pt());
        h_ScoutingMuonNoVtx_eta_->Fill(mu.eta());
        h_ScoutingMuonNoVtx_phi_->Fill(mu.phi());
        h_ScoutingMuonNoVtx_charge_->Fill(mu.charge());
        h_ScoutingMuonNoVtx_trkchi2_->Fill(mu.trk_chi2());
        h_ScoutingMuonNoVtx_trkndof_->Fill(mu.trk_ndof());
        h_ScoutingMuonNoVtx_trkdxy_->Fill(mu.trk_dxy());
        h_ScoutingMuonNoVtx_trkdz_->Fill(mu.trk_dz());
        h_ScoutingMuonNoVtx_trkqoverp_->Fill(mu.trk_qoverp());
        h_ScoutingMuonNoVtx_trklambda_->Fill(mu.trk_lambda());
        h_ScoutingMuonNoVtx_trkpt_->Fill(mu.trk_pt());
        h_ScoutingMuonNoVtx_trkphi_->Fill(mu.trk_phi());
        h_ScoutingMuonNoVtx_trketa_->Fill(mu.trk_eta());
        h_ScoutingMuonNoVtx_trkqoverpError_->Fill(mu.trk_qoverpError());
        h_ScoutingMuonNoVtx_trklambdaError_->Fill(mu.trk_lambdaError());
        h_ScoutingMuonNoVtx_trkdxyError_->Fill(mu.trk_dxyError());
        h_ScoutingMuonNoVtx_trkdzError_->Fill(mu.trk_dzError());
        h_ScoutingMuonNoVtx_trkphiError_->Fill(mu.trk_phiError());
        h_ScoutingMuonNoVtx_trkdsz_->Fill(mu.trk_dsz());
        h_ScoutingMuonNoVtx_trkdszError_->Fill(mu.trk_dszError());
        h_ScoutingMuonNoVtx_trkvx_->Fill(mu.trk_vx());
        h_ScoutingMuonNoVtx_trkvy_->Fill(mu.trk_vy());
        h_ScoutingMuonNoVtx_trkvz_->Fill(mu.trk_vz());
      }

      h_ScoutingMuonNoVtx_m_->Fill(mu.m());
      h_ScoutingMuonNoVtx_vtxIndx_->Fill(mu.vtxIndx().size());
      muonNoVtx_vtxIndx.push_back(mu.vtxIndx());
      // Extrapolated phiCorr logic
      float phiCorr = mu.phi();
      if (SVNoVtx.isValid() && !mu.vtxIndx().empty() && theB) {
        int vtxIndx = mu.vtxIndx()[0];
        if (vtxIndx >= 0 && vtxIndx < int(SVNoVtx->size())) {
          const auto& sv = SVNoVtx->at(vtxIndx);
          reco::Track::Point v(mu.trk_vx(), mu.trk_vy(), mu.trk_vz());
          reco::Track::Vector p(mu.trk_pt() * std::cos(mu.trk_phi()),
                                mu.trk_pt() * std::sin(mu.trk_phi()),
                                mu.trk_pt() * std::sinh(mu.trk_eta()));
          double vec[15];
          for (int k = 0; k < 15; k++)
            vec[k] = 1.;
          reco::TrackBase::CovarianceMatrix cov(vec, vec + 15);
          cov(0, 0) = std::pow(mu.trk_qoverpError(), 2);
          cov(1, 1) = std::pow(mu.trk_lambdaError(), 2);
          cov(2, 2) = std::pow(mu.trk_phiError(), 2);
          cov(3, 3) = std::pow(mu.trk_dxyError(), 2);
          cov(4, 4) = std::pow(mu.trk_dszError(), 2);
          reco::Track trk(mu.trk_chi2(), mu.trk_ndof(), v, p, mu.charge(), cov);
          reco::TransientTrack trans = theB->build(trk);
          GlobalPoint svPos(sv.x(), sv.y(), sv.z());
          auto traj = trans.trajectoryStateClosestToPoint(svPos);
          phiCorr = traj.momentum().phi();
        }
      }
      muonNoVtx_phiCorr.push_back(phiCorr);
      h_ScoutingMuonNoVtx_phiCorr_->Fill(phiCorr);
    }
  }

  // Fill ScoutingMuonVtx block, including phiCorr logic
  std::vector<float> muonVtx_phiCorr;
  std::vector<std::vector<int>> muonVtx_vtxIndx;
  if (muonsVtx.isValid()) {
    if (fillAllHistograms_) {
      h_nScoutingMuonVtx_->Fill(muonsVtx->size());
    }
    for (size_t i = 0; i < muonsVtx->size(); ++i) {
      const auto& mu = muonsVtx->at(i);
      if (fillAllHistograms_) {
        h_ScoutingMuonVtx_pt_->Fill(mu.pt());
        h_ScoutingMuonVtx_eta_->Fill(mu.eta());
        h_ScoutingMuonVtx_phi_->Fill(mu.phi());
        h_ScoutingMuonVtx_charge_->Fill(mu.charge());
        h_ScoutingMuonVtx_trkchi2_->Fill(mu.trk_chi2());
        h_ScoutingMuonVtx_trkndof_->Fill(mu.trk_ndof());
        h_ScoutingMuonVtx_trkdxy_->Fill(mu.trk_dxy());
        h_ScoutingMuonVtx_trkdz_->Fill(mu.trk_dz());
        h_ScoutingMuonVtx_trkqoverp_->Fill(mu.trk_qoverp());
        h_ScoutingMuonVtx_trklambda_->Fill(mu.trk_lambda());
        h_ScoutingMuonVtx_trkpt_->Fill(mu.trk_pt());
        h_ScoutingMuonVtx_trkphi_->Fill(mu.trk_phi());
        h_ScoutingMuonVtx_trketa_->Fill(mu.trk_eta());
        h_ScoutingMuonVtx_trkqoverpError_->Fill(mu.trk_qoverpError());
        h_ScoutingMuonVtx_trklambdaError_->Fill(mu.trk_lambdaError());
        h_ScoutingMuonVtx_trkdxyError_->Fill(mu.trk_dxyError());
        h_ScoutingMuonVtx_trkdzError_->Fill(mu.trk_dzError());
        h_ScoutingMuonVtx_trkphiError_->Fill(mu.trk_phiError());
        h_ScoutingMuonVtx_trkdsz_->Fill(mu.trk_dsz());
        h_ScoutingMuonVtx_trkdszError_->Fill(mu.trk_dszError());
        h_ScoutingMuonVtx_trkvx_->Fill(mu.trk_vx());
        h_ScoutingMuonVtx_trkvy_->Fill(mu.trk_vy());
        h_ScoutingMuonVtx_trkvz_->Fill(mu.trk_vz());
      }

      h_ScoutingMuonVtx_m_->Fill(mu.m());
      h_ScoutingMuonVtx_vtxIndx_->Fill(mu.vtxIndx().size());
      muonVtx_vtxIndx.push_back(mu.vtxIndx());
      // Extrapolated phiCorr logic
      float phiCorr = mu.phi();
      if (SVVtx.isValid() && !mu.vtxIndx().empty() && theB) {
        int vtxIndx = mu.vtxIndx()[0];
        if (vtxIndx >= 0 && vtxIndx < int(SVVtx->size())) {
          const auto& sv = SVVtx->at(vtxIndx);
          reco::Track::Point v(mu.trk_vx(), mu.trk_vy(), mu.trk_vz());
          reco::Track::Vector p(mu.trk_pt() * std::cos(mu.trk_phi()),
                                mu.trk_pt() * std::sin(mu.trk_phi()),
                                mu.trk_pt() * std::sinh(mu.trk_eta()));
          double vec[15];
          for (int k = 0; k < 15; k++)
            vec[k] = 1.;
          reco::TrackBase::CovarianceMatrix cov(vec, vec + 15);
          cov(0, 0) = std::pow(mu.trk_qoverpError(), 2);
          cov(1, 1) = std::pow(mu.trk_lambdaError(), 2);
          cov(2, 2) = std::pow(mu.trk_phiError(), 2);
          cov(3, 3) = std::pow(mu.trk_dxyError(), 2);
          cov(4, 4) = std::pow(mu.trk_dszError(), 2);
          reco::Track trk(mu.trk_chi2(), mu.trk_ndof(), v, p, mu.charge(), cov);
          reco::TransientTrack trans = theB->build(trk);
          GlobalPoint svPos(sv.x(), sv.y(), sv.z());
          auto traj = trans.trajectoryStateClosestToPoint(svPos);
          phiCorr = traj.momentum().phi();
        }
      }
      muonVtx_phiCorr.push_back(phiCorr);
      h_ScoutingMuonVtx_phiCorr_->Fill(phiCorr);
    }
  }

  // SVNoVtx block w/ matching and calculations
  VertexDistance3D vdist;
  VertexDistanceXY vdistXY;
  if (SVNoVtx.isValid()) {
    if (fillAllHistograms_) {
      h_nSVNoVtx_->Fill(SVNoVtx->size());
    }
    for (size_t i = 0; i < SVNoVtx->size(); ++i) {
      const auto& sv = SVNoVtx->at(i);
      if (fillAllHistograms_) {
        h_SVNoVtx_x_->Fill(sv.x());
        h_SVNoVtx_y_->Fill(sv.y());
        h_SVNoVtx_z_->Fill(sv.z());
        h_SVNoVtx_xError_->Fill(sv.xError());
        h_SVNoVtx_yError_->Fill(sv.yError());
        h_SVNoVtx_zError_->Fill(sv.zError());
        h_SVNoVtx_trksize_->Fill(sv.tracksSize());
        h_SVNoVtx_chi2_->Fill(sv.chi2());
        h_SVNoVtx_ndof_->Fill(sv.ndof());
        h_SVNoVtx_isvalidvtx_->Fill(sv.isValidVtx());
      }
      // Calculated for PV0:
      if (pvAvailable) {
        Point svPos(sv.x(), sv.y(), sv.z());
        Error3 svErr;
        svErr(0, 0) = std::pow(sv.xError(), 2);
        svErr(1, 1) = std::pow(sv.yError(), 2);
        svErr(2, 2) = std::pow(sv.zError(), 2);
        reco::Vertex svCand(svPos, svErr, sv.chi2(), sv.ndof(), sv.tracksSize());
        Measurement1D dxy = vdistXY.distance(PV0, svCand);
        Measurement1D dlen = vdist.distance(PV0, svCand);
        h_SVNoVtx_dxy_->Fill(dxy.value());
        h_SVNoVtx_dxySig_->Fill(dxy.significance());
        h_SVNoVtx_dlen_->Fill(dlen.value());
        h_SVNoVtx_dlenSig_->Fill(dlen.significance());
      }

      // SV/muon matching for mass/nMuon (NoVtx)
      int nMuonMatch = 0;
      float sv_mass = -1.;
      TLorentzVector sv_p4;
      if (muonsNoVtx.isValid()) {
        for (size_t j = 0; j < muonsNoVtx->size(); ++j) {
          const auto& mu = muonsNoVtx->at(j);
          const auto& vtxIndx = mu.vtxIndx();
          if (std::find(vtxIndx.begin(), vtxIndx.end(), int(i)) != vtxIndx.end()) {
            nMuonMatch++;
            TLorentzVector mu_p4;
            mu_p4.SetPtEtaPhiM(mu.pt(), mu.eta(), muonNoVtx_phiCorr[j], mu.m());
            sv_p4 += mu_p4;
          }
        }
      }
      if (nMuonMatch > 0)
        sv_mass = sv_p4.M();

      h_SVNoVtx_mass_->Fill(sv_mass);
      h_SVNoVtx_mass_JPsi_->Fill(sv_mass);
      h_SVNoVtx_mass_Z_->Fill(sv_mass);
      h_SVNoVtx_nMuon_->Fill(nMuonMatch);
    }
  }

  // SVVtx block w/ matching and calculations
  if (SVVtx.isValid()) {
    if (fillAllHistograms_) {
      h_nSVVtx_->Fill(SVVtx->size());
    }
    for (size_t i = 0; i < SVVtx->size(); ++i) {
      const auto& sv = SVVtx->at(i);
      if (fillAllHistograms_) {
        h_SVVtx_x_->Fill(sv.x());
        h_SVVtx_y_->Fill(sv.y());
        h_SVVtx_z_->Fill(sv.z());
        h_SVVtx_xError_->Fill(sv.xError());
        h_SVVtx_yError_->Fill(sv.yError());
        h_SVVtx_zError_->Fill(sv.zError());
        h_SVVtx_trksize_->Fill(sv.tracksSize());
        h_SVVtx_chi2_->Fill(sv.chi2());
        h_SVVtx_ndof_->Fill(sv.ndof());
        h_SVVtx_isvalidvtx_->Fill(sv.isValidVtx());
      }
      // Calculated for PV0:
      if (pvAvailable) {
        Point svPos(sv.x(), sv.y(), sv.z());
        Error3 svErr;
        svErr(0, 0) = std::pow(sv.xError(), 2);
        svErr(1, 1) = std::pow(sv.yError(), 2);
        svErr(2, 2) = std::pow(sv.zError(), 2);
        reco::Vertex svCand(svPos, svErr, sv.chi2(), sv.ndof(), sv.tracksSize());
        Measurement1D dxy = vdistXY.distance(PV0, svCand);
        Measurement1D dlen = vdist.distance(PV0, svCand);
        h_SVVtx_dxy_->Fill(dxy.value());
        h_SVVtx_dxySig_->Fill(dxy.significance());
        h_SVVtx_dlen_->Fill(dlen.value());
        h_SVVtx_dlenSig_->Fill(dlen.significance());
      }

      // SV/muon matching for mass/nMuon (Vtx)
      int nMuonMatch = 0;
      float sv_mass = -1.;
      TLorentzVector sv_p4;
      if (muonsVtx.isValid()) {
        for (size_t j = 0; j < muonsVtx->size(); ++j) {
          const auto& mu = muonsVtx->at(j);
          const auto& vtxIndx = mu.vtxIndx();
          if (std::find(vtxIndx.begin(), vtxIndx.end(), int(i)) != vtxIndx.end()) {
            nMuonMatch++;
            TLorentzVector mu_p4;
            mu_p4.SetPtEtaPhiM(mu.pt(), mu.eta(), muonVtx_phiCorr[j], mu.m());
            sv_p4 += mu_p4;
          }
        }
      }
      if (nMuonMatch > 0)
        sv_mass = sv_p4.M();
      h_SVVtx_mass_->Fill(sv_mass);
      h_SVVtx_mass_JPsi_->Fill(sv_mass);
      h_SVVtx_mass_Z_->Fill(sv_mass);
      h_SVVtx_nMuon_->Fill(nMuonMatch);
    }
  }
}

void ScoutingMuonPropertiesAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("OutputInternalPath", "HLT/ScoutingOffline/Muons/Properties");
  desc.add<bool>("fillAllHistograms", false);
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("muonsNoVtx", edm::InputTag("hltScoutingMuonPackerNoVtx"));
  desc.add<edm::InputTag>("muonsVtx", edm::InputTag("hltScoutingMuonPackerVtx"));
  desc.add<edm::InputTag>("PV", edm::InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"));
  desc.add<edm::InputTag>("SVNoVtx", edm::InputTag("hltScoutingMuonPackerNoVtx", "displacedVtx"));
  desc.add<edm::InputTag>("SVVtx", edm::InputTag("hltScoutingMuonPackerVtx", "displacedVtx"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(ScoutingMuonPropertiesAnalyzer);
