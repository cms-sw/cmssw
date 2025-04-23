// -*- C++ -*-
//
// Package:    HLTriggerOffline/Scouting
// Class:      ScoutingCollectionMonitor
//
/**\class ScoutingCollectionMonitor ScoutingCollectionMonitor.cc 
HLTriggerOffline/Scouting/plugins/ScoutingCollectionMonitor.cc

Description: ScoutingCollectionMonitor is developed to enable monitoring of several scouting objects and comparisons for the NGT demonstrator

It is based on the preexisting work of the scouting group and can be found at git@github.com:CMS-Run3ScoutingTools/Run3ScoutingAnalysisTools.git

*/
//
// Original Author:  Jessica Prendi
//         Created:  Thu, 17 Apr 2025 14:15:08 GMT
//
//

// system include files
#include <memory>
#include <TLorentzVector.h>

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingParticle.h"

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//
// class declaration
//

class ScoutingCollectionMonitor : public DQMEDAnalyzer {
public:
  explicit ScoutingCollectionMonitor(const edm::ParameterSet&);
  ~ScoutingCollectionMonitor() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  const std::string outputInternalPath_ = "HLT/ScoutingOffline/Miscellaneous";

  const edm::InputTag triggerResultsTag;
  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken;
  const edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> muonsToken;
  const edm::EDGetTokenT<std::vector<Run3ScoutingElectron>> electronsToken;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> primaryVerticesToken;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> verticesToken;
  const edm::EDGetTokenT<std::vector<Run3ScoutingPhoton>> photonsToken;
  const edm::EDGetTokenT<double> rhoToken;
  const edm::EDGetTokenT<double> pfMetPhiToken;
  const edm::EDGetTokenT<double> pfMetPtToken;
  const edm::EDGetTokenT<std::vector<Run3ScoutingParticle>> pfcandsToken;
  const edm::EDGetTokenT<std::vector<Run3ScoutingPFJet>> pfjetsToken;
  const edm::EDGetTokenT<std::vector<Run3ScoutingTrack>> tracksToken;

  std::vector<std::string> triggerPathsVector;
  std::map<std::string, int> triggerPathsMap;

  bool doL1;
  triggerExpression::Data triggerCache_;

  edm::InputTag algInputTag_;
  edm::InputTag extInputTag_;
  edm::EDGetToken algToken_;

  // rho + pfMetphi + pfMetPt
  dqm::reco::MonitorElement* rho_hist;
  dqm::reco::MonitorElement* pfMetPhi_hist;
  dqm::reco::MonitorElement* pfMetPt_hist;

  // PF candidates histograms
  dqm::reco::MonitorElement* PF_pT_211_hist;
  dqm::reco::MonitorElement* PF_pT_n211_hist;
  dqm::reco::MonitorElement* PF_pT_130_hist;
  dqm::reco::MonitorElement* PF_pT_22_hist;
  dqm::reco::MonitorElement* PF_pT_13_hist;
  dqm::reco::MonitorElement* PF_pT_n13_hist;
  dqm::reco::MonitorElement* PF_pT_1_hist;
  dqm::reco::MonitorElement* PF_pT_2_hist;

  dqm::reco::MonitorElement* PF_eta_211_hist;
  dqm::reco::MonitorElement* PF_eta_n211_hist;
  dqm::reco::MonitorElement* PF_eta_130_hist;
  dqm::reco::MonitorElement* PF_eta_22_hist;
  dqm::reco::MonitorElement* PF_eta_13_hist;
  dqm::reco::MonitorElement* PF_eta_n13_hist;
  dqm::reco::MonitorElement* PF_eta_1_hist;
  dqm::reco::MonitorElement* PF_eta_2_hist;

  dqm::reco::MonitorElement* PF_phi_211_hist;
  dqm::reco::MonitorElement* PF_phi_n211_hist;
  dqm::reco::MonitorElement* PF_phi_130_hist;
  dqm::reco::MonitorElement* PF_phi_22_hist;
  dqm::reco::MonitorElement* PF_phi_13_hist;
  dqm::reco::MonitorElement* PF_phi_n13_hist;
  dqm::reco::MonitorElement* PF_phi_1_hist;
  dqm::reco::MonitorElement* PF_phi_2_hist;

  dqm::reco::MonitorElement* PF_vertex_211_hist;
  dqm::reco::MonitorElement* PF_vertex_n211_hist;
  dqm::reco::MonitorElement* PF_vertex_130_hist;
  dqm::reco::MonitorElement* PF_vertex_22_hist;
  dqm::reco::MonitorElement* PF_vertex_13_hist;
  dqm::reco::MonitorElement* PF_vertex_n13_hist;
  dqm::reco::MonitorElement* PF_vertex_1_hist;
  dqm::reco::MonitorElement* PF_vertex_2_hist;
  dqm::reco::MonitorElement* PF_normchi2_211_hist;
  dqm::reco::MonitorElement* PF_normchi2_n211_hist;
  dqm::reco::MonitorElement* PF_normchi2_130_hist;
  dqm::reco::MonitorElement* PF_normchi2_22_hist;
  dqm::reco::MonitorElement* PF_normchi2_13_hist;
  dqm::reco::MonitorElement* PF_normchi2_n13_hist;
  dqm::reco::MonitorElement* PF_normchi2_1_hist;
  dqm::reco::MonitorElement* PF_normchi2_2_hist;

  dqm::reco::MonitorElement* PF_dz_211_hist;
  dqm::reco::MonitorElement* PF_dz_n211_hist;
  dqm::reco::MonitorElement* PF_dz_13_hist;
  dqm::reco::MonitorElement* PF_dz_n13_hist;

  dqm::reco::MonitorElement* PF_dxy_211_hist;
  dqm::reco::MonitorElement* PF_dxy_n211_hist;
  dqm::reco::MonitorElement* PF_dxy_13_hist;
  dqm::reco::MonitorElement* PF_dxy_n13_hist;

  dqm::reco::MonitorElement* PF_dzsig_211_hist;
  dqm::reco::MonitorElement* PF_dzsig_n211_hist;
  dqm::reco::MonitorElement* PF_dzsig_13_hist;
  dqm::reco::MonitorElement* PF_dzsig_n13_hist;

  dqm::reco::MonitorElement* PF_dxysig_211_hist;
  dqm::reco::MonitorElement* PF_dxysig_n211_hist;
  dqm::reco::MonitorElement* PF_dxysig_13_hist;
  dqm::reco::MonitorElement* PF_dxysig_n13_hist;

  dqm::reco::MonitorElement* PF_trk_pt_211_hist;
  dqm::reco::MonitorElement* PF_trk_pt_n211_hist;
  dqm::reco::MonitorElement* PF_trk_pt_13_hist;
  dqm::reco::MonitorElement* PF_trk_pt_n13_hist;

  dqm::reco::MonitorElement* PF_trk_eta_211_hist;
  dqm::reco::MonitorElement* PF_trk_eta_n211_hist;
  dqm::reco::MonitorElement* PF_trk_eta_13_hist;
  dqm::reco::MonitorElement* PF_trk_eta_n13_hist;

  dqm::reco::MonitorElement* PF_trk_phi_211_hist;
  dqm::reco::MonitorElement* PF_trk_phi_n211_hist;
  dqm::reco::MonitorElement* PF_trk_phi_13_hist;
  dqm::reco::MonitorElement* PF_trk_phi_n13_hist;

  // photon histograms
  dqm::reco::MonitorElement* pt_pho_hist;
  dqm::reco::MonitorElement* eta_pho_hist;
  dqm::reco::MonitorElement* phi_pho_hist;
  dqm::reco::MonitorElement* rawEnergy_pho_hist;
  dqm::reco::MonitorElement* preshowerEnergy_pho_hist;
  dqm::reco::MonitorElement* corrEcalEnergyError_pho_hist;
  dqm::reco::MonitorElement* sigmaIetaIeta_pho_hist;
  dqm::reco::MonitorElement* hOverE_pho_hist;
  dqm::reco::MonitorElement* ecalIso_pho_hist;
  dqm::reco::MonitorElement* hcalIso_pho_hist;
  dqm::reco::MonitorElement* trackIso_pho_hist;
  dqm::reco::MonitorElement* r9_pho_hist;
  dqm::reco::MonitorElement* sMin_pho_hist;
  dqm::reco::MonitorElement* sMaj_pho_hist;

  // electron histograms
  dqm::reco::MonitorElement* pt_ele_hist;
  dqm::reco::MonitorElement* eta_ele_hist;
  dqm::reco::MonitorElement* phi_ele_hist;
  dqm::reco::MonitorElement* rawEnergy_ele_hist;
  dqm::reco::MonitorElement* preshowerEnergy_ele_hist;
  dqm::reco::MonitorElement* corrEcalEnergyError_ele_hist;
  dqm::reco::MonitorElement* dEtaIn_ele_hist;
  dqm::reco::MonitorElement* dPhiIn_ele_hist;
  dqm::reco::MonitorElement* sigmaIetaIeta_ele_hist;
  dqm::reco::MonitorElement* hOverE_ele_hist;
  dqm::reco::MonitorElement* ooEMOop_ele_hist;
  dqm::reco::MonitorElement* missingHits_ele_hist;
  dqm::reco::MonitorElement* trackfbrem_ele_hist;
  dqm::reco::MonitorElement* ecalIso_ele_hist;
  dqm::reco::MonitorElement* hcalIso_ele_hist;
  dqm::reco::MonitorElement* trackIso_ele_hist;
  dqm::reco::MonitorElement* r9_ele_hist;
  dqm::reco::MonitorElement* sMin_ele_hist;
  dqm::reco::MonitorElement* sMaj_ele_hist;

  // muon histograms

  dqm::reco::MonitorElement* pt_mu_hist;
  dqm::reco::MonitorElement* eta_mu_hist;
  dqm::reco::MonitorElement* phi_mu_hist;
  dqm::reco::MonitorElement* type_mu_hist;
  dqm::reco::MonitorElement* charge_mu_hist;
  dqm::reco::MonitorElement* normalizedChi2_mu_hist;
  dqm::reco::MonitorElement* ecalIso_mu_hist;
  dqm::reco::MonitorElement* hcalIso_mu_hist;
  dqm::reco::MonitorElement* trackIso_mu_hist;
  dqm::reco::MonitorElement* nValidStandAloneMuonHits_mu_hist;
  dqm::reco::MonitorElement* nStandAloneMuonMatchedStations_mu_hist;
  dqm::reco::MonitorElement* nValidRecoMuonHits_mu_hist;
  dqm::reco::MonitorElement* nRecoMuonChambers_mu_hist;
  dqm::reco::MonitorElement* nRecoMuonChambersCSCorDT_mu_hist;
  dqm::reco::MonitorElement* nRecoMuonMatches_mu_hist;
  dqm::reco::MonitorElement* nRecoMuonMatchedStations_mu_hist;
  dqm::reco::MonitorElement* nRecoMuonExpectedMatchedStations_mu_hist;
  dqm::reco::MonitorElement* recoMuonStationMask_mu_hist;
  dqm::reco::MonitorElement* nRecoMuonMatchedRPCLayers_mu_hist;
  dqm::reco::MonitorElement* recoMuonRPClayerMask_mu_hist;
  dqm::reco::MonitorElement* nValidPixelHits_mu_hist;
  dqm::reco::MonitorElement* nValidStripHits_mu_hist;
  dqm::reco::MonitorElement* nPixelLayersWithMeasurement_mu_hist;
  dqm::reco::MonitorElement* nTrackerLayersWithMeasurement_mu_hist;
  dqm::reco::MonitorElement* trk_chi2_mu_hist;
  dqm::reco::MonitorElement* trk_ndof_mu_hist;
  dqm::reco::MonitorElement* trk_dxy_mu_hist;
  dqm::reco::MonitorElement* trk_dz_mu_hist;
  dqm::reco::MonitorElement* trk_qoverp_mu_hist;
  dqm::reco::MonitorElement* trk_lambda_mu_hist;
  dqm::reco::MonitorElement* trk_pt_mu_hist;
  dqm::reco::MonitorElement* trk_phi_mu_hist;
  dqm::reco::MonitorElement* trk_eta_mu_hist;
  dqm::reco::MonitorElement* trk_dxyError_mu_hist;
  dqm::reco::MonitorElement* trk_dzError_mu_hist;
  dqm::reco::MonitorElement* trk_qoverpError_mu_hist;
  dqm::reco::MonitorElement* trk_lambdaError_mu_hist;
  dqm::reco::MonitorElement* trk_phiError_mu_hist;
  dqm::reco::MonitorElement* trk_dsz_mu_hist;
  dqm::reco::MonitorElement* trk_dszError_mu_hist;
  dqm::reco::MonitorElement* trk_qoverp_lambda_cov_mu_hist;
  dqm::reco::MonitorElement* trk_qoverp_phi_cov_mu_hist;
  dqm::reco::MonitorElement* trk_qoverp_dxy_cov_mu_hist;
  dqm::reco::MonitorElement* trk_qoverp_dsz_cov_mu_hist;
  dqm::reco::MonitorElement* trk_lambda_phi_cov_mu_hist;
  dqm::reco::MonitorElement* trk_lambda_dxy_cov_mu_hist;
  dqm::reco::MonitorElement* trk_lambda_dsz_cov_mu_hist;
  dqm::reco::MonitorElement* trk_phi_dxy_cov_mu_hist;
  dqm::reco::MonitorElement* trk_phi_dsz_cov_mu_hist;
  dqm::reco::MonitorElement* trk_dxy_dsz_cov_mu_hist;
  dqm::reco::MonitorElement* trk_vx_mu_hist;
  dqm::reco::MonitorElement* trk_vy_mu_hist;
  dqm::reco::MonitorElement* trk_vz_mu_hist;

  // PF Jet histograms
  dqm::reco::MonitorElement* pt_pfj_hist;
  dqm::reco::MonitorElement* eta_pfj_hist;
  dqm::reco::MonitorElement* phi_pfj_hist;
  dqm::reco::MonitorElement* m_pfj_hist;
  dqm::reco::MonitorElement* jetArea_pfj_hist;
  dqm::reco::MonitorElement* chargedHadronEnergy_pfj_hist;
  dqm::reco::MonitorElement* neutralHadronEnergy_pfj_hist;
  dqm::reco::MonitorElement* photonEnergy_pfj_hist;
  dqm::reco::MonitorElement* electronEnergy_pfj_hist;
  dqm::reco::MonitorElement* muonEnergy_pfj_hist;
  dqm::reco::MonitorElement* HFHadronEnergy_pfj_hist;
  dqm::reco::MonitorElement* HFEMEnergy_pfj_hist;
  dqm::reco::MonitorElement* chargedHadronMultiplicity_pfj_hist;
  dqm::reco::MonitorElement* neutralHadronMultiplicity_pfj_hist;
  dqm::reco::MonitorElement* photonMultiplicity_pfj_hist;
  dqm::reco::MonitorElement* electronMultiplicity_pfj_hist;
  dqm::reco::MonitorElement* muonMultiplicity_pfj_hist;
  dqm::reco::MonitorElement* HFHadronMultiplicity_pfj_hist;
  dqm::reco::MonitorElement* HFEMMultiplicity_pfj_hist;
  dqm::reco::MonitorElement* HOEnergy_pfj_hist;
  dqm::reco::MonitorElement* csv_pfj_hist;
  dqm::reco::MonitorElement* mvaDiscriminator_pfj_hist;

  // primary vertex histograms
  dqm::reco::MonitorElement* x_pv_hist;
  dqm::reco::MonitorElement* y_pv_hist;
  dqm::reco::MonitorElement* z_pv_hist;
  dqm::reco::MonitorElement* zError_pv_hist;
  dqm::reco::MonitorElement* xError_pv_hist;
  dqm::reco::MonitorElement* yError_pv_hist;
  dqm::reco::MonitorElement* tracksSize_pv_hist;
  dqm::reco::MonitorElement* chi2_pv_hist;
  dqm::reco::MonitorElement* ndof_pv_hist;
  dqm::reco::MonitorElement* isValidVtx_pv_hist;
  dqm::reco::MonitorElement* xyCov_pv_hist;
  dqm::reco::MonitorElement* xzCov_pv_hist;
  dqm::reco::MonitorElement* yzCov_pv_hist;

  // displaced vertex histograms
  dqm::reco::MonitorElement* x_vtx_hist;
  dqm::reco::MonitorElement* y_vtx_hist;
  dqm::reco::MonitorElement* z_vtx_hist;
  dqm::reco::MonitorElement* zError_vtx_hist;
  dqm::reco::MonitorElement* xError_vtx_hist;
  dqm::reco::MonitorElement* yError_vtx_hist;
  dqm::reco::MonitorElement* tracksSize_vtx_hist;
  dqm::reco::MonitorElement* chi2_vtx_hist;
  dqm::reco::MonitorElement* ndof_vtx_hist;
  dqm::reco::MonitorElement* isValidVtx_vtx_hist;
  dqm::reco::MonitorElement* xyCov_vtx_hist;
  dqm::reco::MonitorElement* xzCov_vtx_hist;
  dqm::reco::MonitorElement* yzCov_vtx_hist;

  // tracker histograms
  dqm::reco::MonitorElement* tk_pt_tk_hist;
  dqm::reco::MonitorElement* tk_eta_tk_hist;
  dqm::reco::MonitorElement* tk_phi_tk_hist;
  dqm::reco::MonitorElement* tk_chi2_tk_hist;
  dqm::reco::MonitorElement* tk_ndof_tk_hist;
  dqm::reco::MonitorElement* tk_charge_tk_hist;
  dqm::reco::MonitorElement* tk_dxy_tk_hist;
  dqm::reco::MonitorElement* tk_dz_tk_hist;
  dqm::reco::MonitorElement* tk_nValidPixelHits_tk_hist;
  dqm::reco::MonitorElement* tk_nTrackerLayersWithMeasurement_tk_hist;
  dqm::reco::MonitorElement* tk_nValidStripHits_tk_hist;
  dqm::reco::MonitorElement* tk_qoverp_tk_hist;
  dqm::reco::MonitorElement* tk_lambda_tk_hist;
  dqm::reco::MonitorElement* tk_dxy_Error_tk_hist;
  dqm::reco::MonitorElement* tk_dz_Error_tk_hist;
  dqm::reco::MonitorElement* tk_qoverp_Error_tk_hist;
  dqm::reco::MonitorElement* tk_lambda_Error_tk_hist;
  dqm::reco::MonitorElement* tk_phi_Error_tk_hist;
  dqm::reco::MonitorElement* tk_dsz_tk_hist;
  dqm::reco::MonitorElement* tk_dsz_Error_tk_hist;
  dqm::reco::MonitorElement* tk_qoverp_lambda_cov_tk_hist;
  dqm::reco::MonitorElement* tk_qoverp_phi_cov_tk_hist;
  dqm::reco::MonitorElement* tk_qoverp_dxy_cov_tk_hist;
  dqm::reco::MonitorElement* tk_qoverp_dsz_cov_tk_hist;
  dqm::reco::MonitorElement* tk_lambda_phi_cov_tk_hist;
  dqm::reco::MonitorElement* tk_lambda_dxy_cov_tk_hist;
  dqm::reco::MonitorElement* tk_lambda_dsz_cov_tk_hist;
  dqm::reco::MonitorElement* tk_phi_dxy_cov_tk_hist;
  dqm::reco::MonitorElement* tk_phi_dsz_cov_tk_hist;
  dqm::reco::MonitorElement* tk_dxy_dsz_cov_tk_hist;
  dqm::reco::MonitorElement* tk_vtxInd_tk_hist;
  dqm::reco::MonitorElement* tk_vx_tk_hist;
  dqm::reco::MonitorElement* tk_vy_tk_hist;
  dqm::reco::MonitorElement* tk_vz_tk_hist;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ScoutingCollectionMonitor::ScoutingCollectionMonitor(const edm::ParameterSet& iConfig)
    : triggerResultsTag(iConfig.getParameter<edm::InputTag>("triggerresults")),
      triggerResultsToken(consumes<edm::TriggerResults>(triggerResultsTag)),
      muonsToken(consumes<std::vector<Run3ScoutingMuon>>(iConfig.getParameter<edm::InputTag>("muons"))),
      electronsToken(consumes<std::vector<Run3ScoutingElectron>>(iConfig.getParameter<edm::InputTag>("electrons"))),
      primaryVerticesToken(
          consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("primaryVertices"))),
      verticesToken(
          consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("displacedVertices"))),
      photonsToken(consumes<std::vector<Run3ScoutingPhoton>>(iConfig.getParameter<edm::InputTag>("photons"))),
      rhoToken(consumes<double>(iConfig.getParameter<edm::InputTag>("rho"))),
      pfMetPhiToken(consumes<double>(iConfig.getParameter<edm::InputTag>("pfMetPhi"))),
      pfMetPtToken(consumes<double>(iConfig.getParameter<edm::InputTag>("pfMetPt"))),
      pfcandsToken(consumes<std::vector<Run3ScoutingParticle>>(iConfig.getParameter<edm::InputTag>("pfcands"))),
      pfjetsToken(consumes<std::vector<Run3ScoutingPFJet>>(iConfig.getParameter<edm::InputTag>("pfjets"))),
      tracksToken(consumes<std::vector<Run3ScoutingTrack>>(iConfig.getParameter<edm::InputTag>("tracks"))) {}

ScoutingCollectionMonitor::~ScoutingCollectionMonitor() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called for each event  ------------
void ScoutingCollectionMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using namespace reco;

  // all the handles needed
  Handle<double> rhoH;
  iEvent.getByToken(rhoToken, rhoH);
  if (!rhoH.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for rho";
    return;
  }

  Handle<double> pfMetPhiH;
  iEvent.getByToken(pfMetPhiToken, pfMetPhiH);
  if (!pfMetPhiH.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for MET phi";
    return;
  }

  Handle<double> pfMetPtH;
  iEvent.getByToken(pfMetPtToken, pfMetPtH);
  if (!pfMetPtH.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for MET pT";
    return;
  }

  Handle<vector<Run3ScoutingParticle>> pfcandsH;
  iEvent.getByToken(pfcandsToken, pfcandsH);
  if (!pfcandsH.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for PF candidates";
    return;
  }

  Handle<vector<Run3ScoutingPhoton>> photonsH;
  iEvent.getByToken(photonsToken, photonsH);
  if (!photonsH.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for photons";
    return;
  }

  Handle<vector<Run3ScoutingElectron>> electronsH;
  iEvent.getByToken(electronsToken, electronsH);
  if (!electronsH.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for electrons";
    return;
  }

  Handle<vector<Run3ScoutingMuon>> muonsH;
  iEvent.getByToken(muonsToken, muonsH);
  if (!muonsH.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for muons";
    return;
  }

  Handle<vector<Run3ScoutingPFJet>> PFjetsH;
  iEvent.getByToken(pfjetsToken, PFjetsH);
  if (!PFjetsH.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for PF jets";
    return;
  }

  Handle<vector<Run3ScoutingVertex>> primaryVerticesH;
  iEvent.getByToken(primaryVerticesToken, primaryVerticesH);
  if (!primaryVerticesH.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for primary vertices";
    return;
  }

  Handle<vector<Run3ScoutingVertex>> verticesH;
  iEvent.getByToken(verticesToken, verticesH);
  if (!verticesH.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for displaced vertices";
    return;
  }

  Handle<vector<Run3ScoutingTrack>> tracksH;
  iEvent.getByToken(tracksToken, tracksH);
  if (!tracksH.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for tracks";
    return;
  }

  // put stuff in histogram
  rho_hist->Fill(*rhoH);
  pfMetPhi_hist->Fill(*pfMetPhiH);
  pfMetPt_hist->Fill(*pfMetPtH);

  // fill the PF candidate histograms (no electrons!)

  for (const auto& cand : *pfcandsH) {
    switch (cand.pdgId()) {
      case 211:
        PF_pT_211_hist->Fill(cand.pt());
        PF_eta_211_hist->Fill(cand.eta());
        PF_phi_211_hist->Fill(cand.phi());
        PF_vertex_211_hist->Fill(cand.vertex());
        PF_normchi2_211_hist->Fill(cand.normchi2());
        PF_dz_211_hist->Fill(cand.dz());
        PF_dxy_211_hist->Fill(cand.dxy());
        PF_dzsig_211_hist->Fill(cand.dzsig());
        PF_dxysig_211_hist->Fill(cand.dxysig());
        PF_trk_pt_211_hist->Fill(cand.trk_pt());
        PF_trk_eta_211_hist->Fill(cand.trk_eta());
        PF_trk_phi_211_hist->Fill(cand.trk_phi());
        break;

      case -211:
        PF_pT_n211_hist->Fill(cand.pt());
        PF_eta_n211_hist->Fill(cand.eta());
        PF_phi_n211_hist->Fill(cand.phi());
        PF_vertex_n211_hist->Fill(cand.vertex());
        PF_normchi2_n211_hist->Fill(cand.normchi2());
        PF_dz_n211_hist->Fill(cand.dz());
        PF_dxy_n211_hist->Fill(cand.dxy());
        PF_dzsig_n211_hist->Fill(cand.dzsig());
        PF_dxysig_n211_hist->Fill(cand.dxysig());
        PF_trk_pt_n211_hist->Fill(cand.trk_pt());
        PF_trk_eta_n211_hist->Fill(cand.trk_eta());
        PF_trk_phi_n211_hist->Fill(cand.trk_phi());
        break;

      case 130:
        PF_pT_130_hist->Fill(cand.pt());
        PF_eta_130_hist->Fill(cand.eta());
        PF_phi_130_hist->Fill(cand.phi());
        PF_vertex_130_hist->Fill(cand.vertex());
        PF_normchi2_130_hist->Fill(cand.normchi2());
        break;

      case 22:
        PF_pT_22_hist->Fill(cand.pt());
        PF_eta_22_hist->Fill(cand.eta());
        PF_phi_22_hist->Fill(cand.phi());
        PF_vertex_22_hist->Fill(cand.vertex());
        PF_normchi2_22_hist->Fill(cand.normchi2());
        break;

      case 13:
        PF_pT_13_hist->Fill(cand.pt());
        PF_eta_13_hist->Fill(cand.eta());
        PF_phi_13_hist->Fill(cand.phi());
        PF_vertex_13_hist->Fill(cand.vertex());
        PF_normchi2_13_hist->Fill(cand.normchi2());
        PF_dz_13_hist->Fill(cand.dz());
        PF_dxy_13_hist->Fill(cand.dxy());
        PF_dzsig_13_hist->Fill(cand.dzsig());
        PF_dxysig_13_hist->Fill(cand.dxysig());
        PF_trk_pt_13_hist->Fill(cand.trk_pt());
        PF_trk_eta_13_hist->Fill(cand.trk_eta());
        PF_trk_phi_13_hist->Fill(cand.trk_phi());
        break;

      case -13:
        PF_pT_n13_hist->Fill(cand.pt());
        PF_eta_n13_hist->Fill(cand.eta());
        PF_phi_n13_hist->Fill(cand.phi());
        PF_vertex_n13_hist->Fill(cand.vertex());
        PF_normchi2_n13_hist->Fill(cand.normchi2());
        PF_dz_n13_hist->Fill(cand.dz());
        PF_dxy_n13_hist->Fill(cand.dxy());
        PF_dzsig_n13_hist->Fill(cand.dzsig());
        PF_dxysig_n13_hist->Fill(cand.dxysig());
        PF_trk_pt_n13_hist->Fill(cand.trk_pt());
        PF_trk_eta_n13_hist->Fill(cand.trk_eta());
        PF_trk_phi_n13_hist->Fill(cand.trk_phi());
        break;

      case 1:
        PF_pT_1_hist->Fill(cand.pt());
        PF_eta_1_hist->Fill(cand.eta());
        PF_phi_1_hist->Fill(cand.phi());
        PF_vertex_1_hist->Fill(cand.vertex());
        PF_normchi2_1_hist->Fill(cand.normchi2());
        break;

      case 2:
        PF_pT_2_hist->Fill(cand.pt());
        PF_eta_2_hist->Fill(cand.eta());
        PF_phi_2_hist->Fill(cand.phi());
        PF_vertex_2_hist->Fill(cand.vertex());
        PF_normchi2_2_hist->Fill(cand.normchi2());
        break;
    }
  }

  // fill all the photon histograms

  for (const auto& pho : *photonsH) {
    pt_pho_hist->Fill(pho.pt());
    eta_pho_hist->Fill(pho.eta());
    phi_pho_hist->Fill(pho.phi());
    rawEnergy_pho_hist->Fill(pho.rawEnergy());
    preshowerEnergy_pho_hist->Fill(pho.preshowerEnergy());
    corrEcalEnergyError_pho_hist->Fill(pho.corrEcalEnergyError());
    sigmaIetaIeta_pho_hist->Fill(pho.sigmaIetaIeta());
    hOverE_pho_hist->Fill(pho.hOverE());
    ecalIso_pho_hist->Fill(pho.ecalIso());
    hcalIso_pho_hist->Fill(pho.hcalIso());
    trackIso_pho_hist->Fill(pho.trkIso());
    r9_pho_hist->Fill(pho.r9());
    sMin_pho_hist->Fill(pho.sMin());
    sMaj_pho_hist->Fill(pho.sMaj());
  }

  // fill all the electron histograms

  for (const auto& ele : *electronsH) {
    pt_ele_hist->Fill(ele.pt());
    eta_ele_hist->Fill(ele.eta());
    phi_ele_hist->Fill(ele.phi());
    rawEnergy_ele_hist->Fill(ele.rawEnergy());
    preshowerEnergy_ele_hist->Fill(ele.preshowerEnergy());
    corrEcalEnergyError_ele_hist->Fill(ele.corrEcalEnergyError());
    dEtaIn_ele_hist->Fill(ele.dEtaIn());
    dPhiIn_ele_hist->Fill(ele.dPhiIn());
    sigmaIetaIeta_ele_hist->Fill(ele.sigmaIetaIeta());
    hOverE_ele_hist->Fill(ele.hOverE());
    ooEMOop_ele_hist->Fill(ele.ooEMOop());
    missingHits_ele_hist->Fill(ele.missingHits());
    trackfbrem_ele_hist->Fill(ele.trackfbrem());
    ecalIso_ele_hist->Fill(ele.ecalIso());
    hcalIso_ele_hist->Fill(ele.hcalIso());
    trackIso_ele_hist->Fill(ele.trackIso());
    r9_ele_hist->Fill(ele.r9());
    sMin_ele_hist->Fill(ele.sMin());
    sMaj_ele_hist->Fill(ele.sMaj());
  }

  // fill all the muon histograms

  for (const auto& mu : *muonsH) {
    pt_mu_hist->Fill(mu.pt());
    eta_mu_hist->Fill(mu.eta());
    phi_mu_hist->Fill(mu.phi());
    type_mu_hist->Fill(mu.type());
    charge_mu_hist->Fill(mu.charge());
    normalizedChi2_mu_hist->Fill(mu.normalizedChi2());
    ecalIso_mu_hist->Fill(mu.ecalIso());
    hcalIso_mu_hist->Fill(mu.hcalIso());
    trackIso_mu_hist->Fill(mu.trackIso());
    nValidStandAloneMuonHits_mu_hist->Fill(mu.nValidStandAloneMuonHits());
    nStandAloneMuonMatchedStations_mu_hist->Fill(mu.nStandAloneMuonMatchedStations());
    nValidRecoMuonHits_mu_hist->Fill(mu.nValidRecoMuonHits());
    nRecoMuonChambers_mu_hist->Fill(mu.nRecoMuonChambers());
    nRecoMuonChambersCSCorDT_mu_hist->Fill(mu.nRecoMuonChambersCSCorDT());
    nRecoMuonMatches_mu_hist->Fill(mu.nRecoMuonMatches());
    nRecoMuonMatchedStations_mu_hist->Fill(mu.nRecoMuonMatchedStations());
    nRecoMuonExpectedMatchedStations_mu_hist->Fill(mu.nRecoMuonExpectedMatchedStations());
    recoMuonStationMask_mu_hist->Fill(mu.recoMuonStationMask());
    nRecoMuonMatchedRPCLayers_mu_hist->Fill(mu.nRecoMuonMatchedRPCLayers());
    recoMuonRPClayerMask_mu_hist->Fill(mu.recoMuonRPClayerMask());
    nValidPixelHits_mu_hist->Fill(mu.nValidPixelHits());
    nValidStripHits_mu_hist->Fill(mu.nValidStripHits());
    nPixelLayersWithMeasurement_mu_hist->Fill(mu.nPixelLayersWithMeasurement());
    nTrackerLayersWithMeasurement_mu_hist->Fill(mu.nTrackerLayersWithMeasurement());
    trk_chi2_mu_hist->Fill(mu.trk_chi2());
    trk_ndof_mu_hist->Fill(mu.trk_ndof());
    trk_dxy_mu_hist->Fill(mu.trk_dxy());
    trk_dz_mu_hist->Fill(mu.trk_dz());
    trk_qoverp_mu_hist->Fill(mu.trk_qoverp());
    trk_lambda_mu_hist->Fill(mu.trk_lambda());
    trk_pt_mu_hist->Fill(mu.trk_pt());
    trk_phi_mu_hist->Fill(mu.trk_phi());
    trk_eta_mu_hist->Fill(mu.trk_eta());
    trk_dxyError_mu_hist->Fill(mu.trk_dxyError());
    trk_dzError_mu_hist->Fill(mu.trk_dzError());
    trk_qoverpError_mu_hist->Fill(mu.trk_qoverpError());
    trk_lambdaError_mu_hist->Fill(mu.trk_lambdaError());
    trk_phiError_mu_hist->Fill(mu.trk_phiError());
    trk_dsz_mu_hist->Fill(mu.trk_dsz());
    trk_dszError_mu_hist->Fill(mu.trk_dszError());
    trk_qoverp_lambda_cov_mu_hist->Fill(mu.trk_qoverp_lambda_cov());
    trk_qoverp_phi_cov_mu_hist->Fill(mu.trk_qoverp_phi_cov());
    trk_qoverp_dxy_cov_mu_hist->Fill(mu.trk_qoverp_dxy_cov());
    trk_qoverp_dsz_cov_mu_hist->Fill(mu.trk_qoverp_dsz_cov());
    trk_lambda_phi_cov_mu_hist->Fill(mu.trk_lambda_phi_cov());
    trk_lambda_dxy_cov_mu_hist->Fill(mu.trk_lambda_dxy_cov());
    trk_lambda_dsz_cov_mu_hist->Fill(mu.trk_lambda_dsz_cov());
    trk_phi_dxy_cov_mu_hist->Fill(mu.trk_phi_dxy_cov());
    trk_phi_dsz_cov_mu_hist->Fill(mu.trk_phi_dsz_cov());
    trk_dxy_dsz_cov_mu_hist->Fill(mu.trk_dxy_dsz_cov());
    trk_vx_mu_hist->Fill(mu.trk_vx());
    trk_vy_mu_hist->Fill(mu.trk_vy());
    trk_vz_mu_hist->Fill(mu.trk_vz());
  }

  // fill all the PF Jet histograms
  for (const auto& jet : *PFjetsH) {
    pt_pfj_hist->Fill(jet.pt());
    eta_pfj_hist->Fill(jet.eta());
    phi_pfj_hist->Fill(jet.phi());
    m_pfj_hist->Fill(jet.m());
    jetArea_pfj_hist->Fill(jet.jetArea());
    chargedHadronEnergy_pfj_hist->Fill(jet.chargedHadronEnergy());
    neutralHadronEnergy_pfj_hist->Fill(jet.neutralHadronEnergy());
    photonEnergy_pfj_hist->Fill(jet.photonEnergy());
    electronEnergy_pfj_hist->Fill(jet.electronEnergy());
    muonEnergy_pfj_hist->Fill(jet.muonEnergy());
    HFHadronEnergy_pfj_hist->Fill(jet.HFHadronEnergy());
    HFEMEnergy_pfj_hist->Fill(jet.HFEMEnergy());
    chargedHadronMultiplicity_pfj_hist->Fill(jet.chargedHadronMultiplicity());
    neutralHadronMultiplicity_pfj_hist->Fill(jet.neutralHadronMultiplicity());
    photonMultiplicity_pfj_hist->Fill(jet.photonMultiplicity());
    electronMultiplicity_pfj_hist->Fill(jet.electronMultiplicity());
    muonMultiplicity_pfj_hist->Fill(jet.muonMultiplicity());
    HFHadronMultiplicity_pfj_hist->Fill(jet.HFHadronMultiplicity());
    HFEMMultiplicity_pfj_hist->Fill(jet.HFEMMultiplicity());
    HOEnergy_pfj_hist->Fill(jet.HOEnergy());
    csv_pfj_hist->Fill(jet.csv());
    mvaDiscriminator_pfj_hist->Fill(jet.mvaDiscriminator());
  }

  // fill all the primary vertices histograms
  for (const auto& vtx : *primaryVerticesH) {
    x_pv_hist->Fill(vtx.x());
    y_pv_hist->Fill(vtx.y());
    z_pv_hist->Fill(vtx.z());
    zError_pv_hist->Fill(vtx.zError());
    xError_pv_hist->Fill(vtx.xError());
    yError_pv_hist->Fill(vtx.yError());
    tracksSize_pv_hist->Fill(vtx.tracksSize());
    chi2_pv_hist->Fill(vtx.chi2());
    ndof_pv_hist->Fill(vtx.ndof());
    isValidVtx_pv_hist->Fill(vtx.isValidVtx());
    xyCov_pv_hist->Fill(vtx.xyCov());
    xzCov_pv_hist->Fill(vtx.xzCov());
    yzCov_pv_hist->Fill(vtx.yzCov());
  }

  // fill all the displaced vertices histograms
  for (const auto& vtx : *verticesH) {
    x_vtx_hist->Fill(vtx.x());
    y_vtx_hist->Fill(vtx.y());
    z_vtx_hist->Fill(vtx.z());
    zError_vtx_hist->Fill(vtx.zError());
    xError_vtx_hist->Fill(vtx.xError());
    yError_vtx_hist->Fill(vtx.yError());
    tracksSize_vtx_hist->Fill(vtx.tracksSize());
    chi2_vtx_hist->Fill(vtx.chi2());
    ndof_vtx_hist->Fill(vtx.ndof());
    isValidVtx_vtx_hist->Fill(vtx.isValidVtx());
    xyCov_vtx_hist->Fill(vtx.xyCov());
    xzCov_vtx_hist->Fill(vtx.xzCov());
    yzCov_vtx_hist->Fill(vtx.yzCov());
  }

  // fill tracks histograms
  for (const auto& tk : *tracksH) {
    tk_pt_tk_hist->Fill(tk.tk_pt());
    tk_eta_tk_hist->Fill(tk.tk_eta());
    tk_phi_tk_hist->Fill(tk.tk_phi());
    tk_chi2_tk_hist->Fill(tk.tk_chi2());
    tk_ndof_tk_hist->Fill(tk.tk_ndof());
    tk_charge_tk_hist->Fill(tk.tk_charge());
    tk_dxy_tk_hist->Fill(tk.tk_dxy());
    tk_dz_tk_hist->Fill(tk.tk_dz());
    tk_nValidPixelHits_tk_hist->Fill(tk.tk_nValidPixelHits());
    tk_nTrackerLayersWithMeasurement_tk_hist->Fill(tk.tk_nTrackerLayersWithMeasurement());
    tk_nValidStripHits_tk_hist->Fill(tk.tk_nValidStripHits());
    tk_qoverp_tk_hist->Fill(tk.tk_qoverp());
    tk_lambda_tk_hist->Fill(tk.tk_lambda());
    tk_dxy_Error_tk_hist->Fill(tk.tk_dxy_Error());
    tk_dz_Error_tk_hist->Fill(tk.tk_dz_Error());
    tk_qoverp_Error_tk_hist->Fill(tk.tk_qoverp_Error());
    tk_lambda_Error_tk_hist->Fill(tk.tk_lambda_Error());
    tk_phi_Error_tk_hist->Fill(tk.tk_phi_Error());
    tk_vtxInd_tk_hist->Fill(tk.tk_vtxInd());
    tk_vx_tk_hist->Fill(tk.tk_vx());
    tk_vy_tk_hist->Fill(tk.tk_vy());
    tk_vz_tk_hist->Fill(tk.tk_vz());
  }
}

// ------------ method called once each job just before starting event loop  ------------
void ScoutingCollectionMonitor::bookHistograms(DQMStore::IBooker& ibook,
                                               edm::Run const& run,
                                               edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(outputInternalPath_);
  rho_hist = ibook.book1D("rho", "#rho; #rho; Entries", 100, 0.0, 60.0);
  pfMetPhi_hist = ibook.book1D("pfMetPhi", "pf MET #phi; #phi ;Entries", 100, -3.14, 3.14);
  pfMetPt_hist = ibook.book1D("pfMetPt", "pf MET pT;p_{T} [GeV];Entries", 100, 0.0, 250.0);

  ibook.setCurrentFolder(outputInternalPath_ + "/PFcand");
  PF_pT_211_hist = ibook.book1DD("pT_211", "PF h^{+}  pT (GeV);p_{T} [GeV];Entries", 100, 0.0, 13.0);
  PF_pT_n211_hist = ibook.book1DD("pT_n211", "PF h^{-} pT (GeV);p_{T} [GeV];Entries", 100, 0.0, 14.0);
  PF_pT_130_hist = ibook.book1DD("pT_130", "PF h^{0} pT (GeV);p_{T} [GeV];Entries", 100, 0.0, 20.0);
  PF_pT_22_hist = ibook.book1DD("pT_22", "PF #gamma pT (GeV);p_{T} [GeV];Entries", 100, 0.0, 18.0);
  PF_pT_13_hist = ibook.book1DD("pT_13", "PF #mu^{+} pT (GeV);p_{T} [GeV];Entries", 100, 0.0, 80.0);
  PF_pT_n13_hist = ibook.book1DD("pT_n13", "PF #mu^{-} pT (GeV);p_{T} [GeV];Entries", 100, 0.0, 80.0);
  PF_pT_2_hist = ibook.book1DD("pT_2", "PF HF h (GeV);pT [GeV];Entries", 100, 0.0, 4.5);
  PF_pT_1_hist = ibook.book1DD("pT_1", "PF HF e/#gamma pT (GeV);p_{T} [GeV];Entries", 100, 0.0, 6.0);

  PF_eta_211_hist = ibook.book1DD("eta_211", "PF h^{+} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_n211_hist = ibook.book1DD("eta_n211", "PF h^{-} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_130_hist = ibook.book1DD("eta_130", "PF h^{0} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_22_hist = ibook.book1DD("eta_22", "PF #gamma #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_13_hist = ibook.book1DD("eta_13", "PF #mu^{+} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_n13_hist = ibook.book1DD("eta_n13", "PF #mu^{-} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_1_hist = ibook.book1DD("eta_2", "PF HF h #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_2_hist = ibook.book1DD("eta_1", "PF HF e/#gamma #eta;#eta;Entries", 100, -5.0, 5.0);

  PF_phi_211_hist = ibook.book1DD("phi_211", "PF h^{+} #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_n211_hist = ibook.book1DD("phi_n211", "PF h^{-} #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_130_hist = ibook.book1DD("phi_130", "PF h^{0} #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_22_hist = ibook.book1DD("phi_22", "PF #gamma #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_13_hist = ibook.book1DD("phi_13", "PF #mu^{+} #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_n13_hist = ibook.book1DD("phi_n13", "PF #mu^{-} #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_1_hist = ibook.book1DD("phi_2", "PF HF h #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_2_hist = ibook.book1DD("phi_1", "PF HF e/#gamma #phi;#phi;Entries", 100, -3.2, 3.2);

  PF_vertex_211_hist = ibook.book1DD("vertex_211", "PF h^{+} Vertex;Vertex;Entries", 100, -10.0, 15.0);
  PF_vertex_n211_hist = ibook.book1DD("vertex_n211", "PF h^{-} Vertex;Vertex;Entries", 100, -10.0, 15.0);
  PF_vertex_130_hist = ibook.book1DD("vertex_130", "PF h^{0} Vertex;Vertex;Entries", 100, -10.0, 10.0);
  PF_vertex_22_hist = ibook.book1DD("vertex_22", "PF #gamma Vertex;Vertex;Entries", 100, -10.0, 10.0);
  PF_vertex_13_hist = ibook.book1DD("vertex_13", "PF #mu^{+} Vertex;Vertex;Entries", 100, -10.0, 15.0);
  PF_vertex_n13_hist = ibook.book1DD("vertex_n13", "PF #mu^{-} Vertex;Vertex;Entries", 100, -10.0, 15.0);
  PF_vertex_1_hist = ibook.book1DD("vertex_1", "PF HF h Vertex;Vertex;Entries", 100, -10.0, 10.0);
  PF_vertex_2_hist = ibook.book1DD("vertex_2", "PF HF e/#gamma Vertex;Vertex;Entries", 100, -10.0, 10.0);

  PF_normchi2_211_hist = ibook.book1DD("normchi2_211", "PF h^{+} Norm #chi^2;Norm #chi^2;Entries", 100, 0.0, 10.0);
  PF_normchi2_n211_hist = ibook.book1DD("normchi2_n211", "PF h^{-} Norm #chi^2;Norm #chi^2;Entries", 100, 0.0, 10.0);
  PF_normchi2_130_hist = ibook.book1DD("normchi2_130", "PF h^{0} Norm #chi^2;Norm #chi^2;Entries", 100, 0.0, 100.0);
  PF_normchi2_22_hist = ibook.book1DD("normchi2_22", "PF #gamma Norm #chi^2;Norm #chi^2;Entries", 100, 0.0, 100.0);
  PF_normchi2_13_hist = ibook.book1DD("normchi2_13", "PF #mu^{+} Norm #chi^2;Norm #chi^2;Entries", 100, 0.0, 10.0);
  PF_normchi2_n13_hist = ibook.book1DD("normchi2_n13", "PF #mu^{-} Norm #chi^2;Norm #chi^2;Entries", 100, 0.0, 10.0);
  PF_normchi2_1_hist = ibook.book1DD("normchi2_1", "PF HF h Norm #chi^2;Norm #chi^2;Entries", 100, 0.0, 100.0);
  PF_normchi2_2_hist = ibook.book1DD("normchi2_2", "PF HF e/#gamma Norm #chi^2;Norm #chi^2;Entries", 100, 0.0, 100.0);

  PF_dz_211_hist = ibook.book1DD("dz_211", "PF h^{+} dz (cm);dz (cm);Entries", 100, -1.0, 1.0);
  PF_dz_n211_hist = ibook.book1DD("dz_n211", "PF h^{-} dz (cm);dz (cm);Entries", 100, -1.0, 1.0);
  PF_dz_13_hist = ibook.book1DD("dz_13", "PF #mu^{+} dz (cm);dz (cm);Entries", 100, -1.0, 1.0);
  PF_dz_n13_hist = ibook.book1DD("dz_n13", "PF #mu^{-} dz (cm);dz (cm);Entries", 100, -1.0, 1.0);

  PF_dxy_211_hist = ibook.book1DD("dxy_211", "PF h^{+} dxy (cm);dxy (cm);Entries", 100, -0.5, 0.5);
  PF_dxy_n211_hist = ibook.book1DD("dxy_n211", "PF h^{-} dxy (cm);dxy (cm);Entries", 100, -0.5, 0.5);
  PF_dxy_13_hist = ibook.book1DD("dxy_13", "PF #mu^{+} dxy (cm);dxy (cm);Entries", 100, -0.5, 0.5);
  PF_dxy_n13_hist = ibook.book1DD("dxy_n13", "PF #mu^{-} dxy (cm);dxy (cm);Entries", 100, -0.5, 0.5);

  PF_dzsig_211_hist = ibook.book1DD("dzsig_211", "PF h^{+} dzsig;dzsig;Entries", 100, 0.0, 10.0);
  PF_dzsig_n211_hist = ibook.book1DD("dzsig_n211", "PF h^{-} dzsig;dzsig;Entries", 100, 0.0, 10.0);
  PF_dzsig_13_hist = ibook.book1DD("dzsig_13", "PF #mu^{+} dzsig;dzsig;Entries", 100, 0.0, 10.0);
  PF_dzsig_n13_hist = ibook.book1DD("dzsig_n13", "PF #mu^{-} dzsig;dzsig;Entries", 100, 0.0, 10.0);

  PF_dxysig_211_hist = ibook.book1DD("dxysig_211", "PF h^{+} dxysig;dxysig;Entries", 100, 0.0, 10.0);
  PF_dxysig_n211_hist = ibook.book1DD("dxysig_n211", "PF h^{-} dxysig;dxysig;Entries", 100, 0.0, 10.0);
  PF_dxysig_13_hist = ibook.book1DD("dxysig_13", "PF #mu^{+} dxysig;dxysig;Entries", 100, 0.0, 10.0);
  PF_dxysig_n13_hist = ibook.book1DD("dxysig_n13", "PF #mu^{-} dxysig;dxysig;Entries", 100, 0.0, 10.0);

  PF_trk_pt_211_hist = ibook.book1DD("trk_pt_211", "PF h^{+} Track pT (GeV);Track p_{T} (GeV);Entries", 100, 0.0, 10.0);
  PF_trk_pt_n211_hist =
      ibook.book1DD("trk_pt_n211", "PF h^{-} Track pT (GeV);Track p_{T} (GeV);Entries", 100, 0.0, 10.0);
  PF_trk_pt_13_hist = ibook.book1DD("trk_pt_13", "PF #mu^{+} Track pT (GeV);Track p_{T} (GeV);Entries", 100, 0.0, 10.0);
  PF_trk_pt_n13_hist =
      ibook.book1DD("trk_pt_n13", "PF #mu^{-} Track pT (GeV);Track p_{T} (GeV);Entries", 100, 0.0, 10.0);

  PF_trk_eta_211_hist = ibook.book1DD("trk_eta_211", "PF h^{+} Track #eta;Track #eta;Entries", 100, -3.0, 3.0);
  PF_trk_eta_n211_hist = ibook.book1DD("trk_eta_n211", "PF h^{-} Track #eta;Track #eta;Entries", 100, -3.0, 3.0);
  PF_trk_eta_13_hist = ibook.book1DD("trk_eta_13", "PF #mu^{+} Track #eta;Track #eta;Entries", 100, -3.0, 3.0);
  PF_trk_eta_n13_hist = ibook.book1DD("trk_eta_n13", "PF #mu^{-} Track #eta;Track #eta;Entries", 100, -3.0, 3.0);

  PF_trk_phi_211_hist = ibook.book1DD("trk_phi_211", "PF h^{+} Track #phi;Track #phi;Entries", 100, -3.2, 3.2);
  PF_trk_phi_n211_hist = ibook.book1DD("trk_phi_n211", "PF h^{-} Track #phi;Track #phi;Entries", 100, -3.2, 3.2);
  PF_trk_phi_13_hist = ibook.book1DD("trk_phi_13", "PF #mu^{+} Track #phi;Track #phi;Entries", 100, -3.2, 3.2);
  PF_trk_phi_n13_hist = ibook.book1DD("trk_phi_n13", "PF #mu^{-} Track #phi;Track #phi;Entries", 100, -3.2, 3.2);

  ibook.setCurrentFolder(outputInternalPath_ + "/Photon");
  pt_pho_hist = ibook.book1D("pt_pho", "Photon pT; p_{T} (GeV); Entries", 100, 0.0, 100.0);
  eta_pho_hist = ibook.book1D("eta_pho", "photon #eta; #eta; Entries", 100, -2.7, 2.7);
  phi_pho_hist = ibook.book1D("phi_pho", "Photon #phi; #phi (rad); Entries", 100, -3.14, 3.14);
  rawEnergy_pho_hist = ibook.book1D("rawEnergy_pho", "Raw Energy Photon; Energy (GeV); Entries", 100, 0.0, 250.0);
  preshowerEnergy_pho_hist =
      ibook.book1D("preshowerEnergy_pho", "Preshower Energy Photon; Energy (GeV); Entries", 100, 0.0, 8.0);
  corrEcalEnergyError_pho_hist = ibook.book1D(
      "corrEcalEnergyError_pho", "Corrected ECAL Energy Error Photon; Energy Error (GeV); Entries", 100, 0.0, 20.0);
  sigmaIetaIeta_pho_hist =
      ibook.book1D("sigmaIetaIeta_pho", "Sigma iEta iEta Photon; #sigma_{i#eta i#eta}; Entries", 100, 0.0, 0.5);
  hOverE_pho_hist = ibook.book1D("hOverE_pho", "H/E Photon; H/E; Entries", 100, 0.0, 1.5);
  ecalIso_pho_hist = ibook.book1D("ecalIso_pho", "ECAL Isolation Photon; Isolation (GeV); Entries", 100, 0.0, 100.0);
  hcalIso_pho_hist = ibook.book1D("hcalIso_pho", "HCAL Isolation Photon; Isolation (GeV); Entries", 100, 0.0, 100.0);
  trackIso_pho_hist = ibook.book1D("trackIso_pho", "Track Isolation Photon; Isolation (GeV); Entries", 100, 0.0, 0.05);
  r9_pho_hist = ibook.book1D("r9_pho", "R9; R9; Entries", 100, 0.0, 5);
  sMin_pho_hist = ibook.book1D("sMin_pho", "sMin Photon; sMin; Entries", 100, 0.0, 3);
  sMaj_pho_hist = ibook.book1D("sMaj_pho", "sMaj Photon; sMaj; Entries", 100, 0.0, 3);

  ibook.setCurrentFolder(outputInternalPath_ + "/Electron");
  pt_ele_hist = ibook.book1D("pt_ele", "Electron pT; p_{T} (GeV); Entries", 100, 0.0, 100.0);
  eta_ele_hist = ibook.book1D("eta_ele", "Electron #eta; #eta; Entries", 100, -2.7, 2.7);
  phi_ele_hist = ibook.book1D("phi_ele", "Electron #phi; #phi (rad); Entries", 100, -3.14, 3.14);
  rawEnergy_ele_hist = ibook.book1D("rawEnergy_ele", "Raw Energy Electron; Energy (GeV); Entries", 100, 0.0, 250.0);
  preshowerEnergy_ele_hist =
      ibook.book1D("preshowerEnergy_ele", "Preshower Energy Electron; Energy (GeV); Entries", 100, 0.0, 10.0);
  corrEcalEnergyError_ele_hist = ibook.book1D(
      "corrEcalEnergyError_ele", "Corrected ECAL Energy Error Electron; Energy Error (GeV); Entries", 100, 0.0, 20.0);
  dEtaIn_ele_hist = ibook.book1D("dEtaIn_ele", "dEtaIn Electron; dEtaIn; Entries", 100, -0.05, 0.05);
  dPhiIn_ele_hist = ibook.book1D("dPhiIn_ele", "dPhiIn Electron; dPhiIn; Entries", 100, -0.5, 0.5);
  sigmaIetaIeta_ele_hist =
      ibook.book1D("sigmaIetaIeta_ele", "Sigma iEta iEta Electron; #sigma_{i#eta i#eta}; Entries", 100, 0.0, 0.05);
  hOverE_ele_hist = ibook.book1D("hOverE_ele", "H/E Electron; H/E; Entries", 100, 0.0, 0.3);
  ooEMOop_ele_hist = ibook.book1D("ooEMOop_ele", "1/E - 1/p Electron; 1/E - 1/p; Entries", 100, -0.3, 0.3);
  missingHits_ele_hist = ibook.book1D("missingHits_ele", "Missing Hits Electron; Count; Entries", 10, 0, 5);
  trackfbrem_ele_hist = ibook.book1D("trackfbrem_ele", "Track fbrem Electron; fbrem; Entries", 100, -1.5, 1.0);
  ecalIso_ele_hist = ibook.book1D("ecalIso_ele", "ECAL Isolation Electron; Isolation (GeV); Entries", 100, 0.0, 70.0);
  hcalIso_ele_hist = ibook.book1D("hcalIso_ele", "HCAL Isolation Electron; Isolation (GeV); Entries", 100, 0.0, 60.0);
  trackIso_ele_hist =
      ibook.book1D("trackIso_ele", "Track Isolation Electron; Isolation (GeV); Entries", 100, 0.0, 0.05);
  r9_ele_hist = ibook.book1D("r9_ele", "R9 Electron; R9; Entries", 100, 0.0, 5);
  sMin_ele_hist = ibook.book1D("sMin_ele", "sMin Electron; sMin; Entries", 100, 0.0, 3);
  sMaj_ele_hist = ibook.book1D("sMaj_ele", "sMaj Electron; sMaj; Entries", 100, 0.0, 3);

  ibook.setCurrentFolder(outputInternalPath_ + "/Muon");
  pt_mu_hist = ibook.book1D("pt_mu", "Muon pT; p_{T} (GeV); Entries", 100, 0.0, 200.0);
  eta_mu_hist = ibook.book1D("eta_mu", "Muon #eta; #eta; Entries", 100, -2.7, 2.7);
  phi_mu_hist = ibook.book1D("phi_mu", "Muon #phi; #phi (rad); Entries", 100, -3.14, 3.14);
  type_mu_hist = ibook.book1D("type_mu", "Muon Type; Type; Entries", 10, 0, 10);
  charge_mu_hist = ibook.book1D("charge_mu", "Muon Charge; Charge; Entries", 3, -1, 2);
  normalizedChi2_mu_hist = ibook.book1D("normalizedChi2_mu", "Normalized chi2; chi2; Entries", 100, 0.0, 10.0);
  ecalIso_mu_hist = ibook.book1D("ecalIso_mu", "ECAL Isolation Muon; Isolation (GeV); Entries", 100, 0.0, 100.0);
  hcalIso_mu_hist = ibook.book1D("hcalIso_mu", "HCAL Isolation Muon; Isolation (GeV); Entries", 100, 0.0, 100.0);
  trackIso_mu_hist = ibook.book1D("trackIso_mu", "Track Isolation Muon; Isolation (GeV); Entries", 100, 0.0, 10.0);
  nValidStandAloneMuonHits_mu_hist =
      ibook.book1D("nValidStandAloneMuonHits_mu", "Valid Standalone Muon Hits; Hits; Entries", 50, 0, 50);
  nStandAloneMuonMatchedStations_mu_hist = ibook.book1D(
      "nStandAloneMuonMatchedStations_mu", "Standalone Muon Matched Stations; Stations; Entries", 10, 0, 10);
  nValidRecoMuonHits_mu_hist = ibook.book1D("nValidRecoMuonHits_mu", "Valid Reco Muon Hits; Hits; Entries", 50, 0, 50);
  nRecoMuonChambers_mu_hist = ibook.book1D("nRecoMuonChambers_mu", "Reco Muon Chambers; Chambers; Entries", 10, 0, 20);
  nRecoMuonChambersCSCorDT_mu_hist =
      ibook.book1D("nRecoMuonChambersCSCorDT_mu", "Reco Muon Chambers (CSC or DT); Chambers; Entries", 10, 0, 14);
  nRecoMuonMatches_mu_hist = ibook.book1D("nRecoMuonMatches_mu", "Reco Muon Matches; Matches; Entries", 10, 0, 10);
  nRecoMuonMatchedStations_mu_hist =
      ibook.book1D("nRecoMuonMatchedStations_mu", "Reco Muon Matched Stations; Stations; Entries", 10, 0, 10);
  nRecoMuonExpectedMatchedStations_mu_hist = ibook.book1D(
      "nRecoMuonExpectedMatchedStations_mu", "Reco Muon Expected Matched Stations; Stations; Entries", 10, 0, 10);
  recoMuonStationMask_mu_hist =
      ibook.book1D("recoMuonStationMask_mu", "Reco Muon Station Mask; Mask; Entries", 20, 0, 20);
  nRecoMuonMatchedRPCLayers_mu_hist =
      ibook.book1D("nRecoMuonMatchedRPCLayers_mu", "Reco Muon Matched RPC Layers; Layers; Entries", 10, 0, 2);
  recoMuonRPClayerMask_mu_hist =
      ibook.book1D("recoMuonRPClayerMask_mu", "Reco Muon RPC Layer Mask; Mask; Entries", 20, 0, 5);
  nValidPixelHits_mu_hist = ibook.book1D("nValidPixelHits_mu", "Valid Pixel Hits; Hits; Entries", 20, 0, 20);
  nValidStripHits_mu_hist = ibook.book1D("nValidStripHits_mu", "Valid Strip Hits; Hits; Entries", 50, 0, 50);
  nPixelLayersWithMeasurement_mu_hist =
      ibook.book1D("nPixelLayersWithMeasurement_mu", "Pixel Layers with Measurement; Layers; Entries", 10, 0, 10);
  nTrackerLayersWithMeasurement_mu_hist =
      ibook.book1D("nTrackerLayersWithMeasurement_mu", "Tracker Layers with Measurement; Layers; Entries", 20, 0, 20);
  trk_chi2_mu_hist = ibook.book1D("trk_chi2_mu", "Muon Tracker chi2; #chi^{2}; Entries", 100, 0.0, 100.0);
  trk_ndof_mu_hist = ibook.book1D("trk_ndof_mu", "Muon Tracker Ndof; Ndof; Entries", 100, 0, 100);
  trk_dxy_mu_hist = ibook.book1D("trk_dxy_mu", "Muon Tracker dxy; dxy (cm); Entries", 100, -0.5, 0.5);
  trk_dz_mu_hist = ibook.book1D("trk_dz_mu", "Muon Tracker dz; dz (cm); Entries", 100, -20.0, 20.0);
  trk_qoverp_mu_hist = ibook.book1D("trk_qoverp_mu", "Muon q/p; q/p; Entries", 100, -1, 1);
  trk_lambda_mu_hist = ibook.book1D("trk_lambda_mu", "Muon Lambda; #lambda; Entries", 100, -2, 2);
  trk_pt_mu_hist = ibook.book1D("trk_pt_mu", "Muon Tracker pT; p_{T} (GeV); Entries", 100, 0.0, 200.0);
  trk_phi_mu_hist = ibook.book1D("trk_phi_mu", "Muon Tracker #phi; #phi (rad); Entries", 100, -3.14, 3.14);
  trk_eta_mu_hist = ibook.book1D("trk_eta_mu", "Muon Tracker #eta; #eta; Entries", 100, -2.7, 2.7);
  trk_dxyError_mu_hist = ibook.book1D("trk_dxyError_mu", "Muon dxy Error; dxy Error (cm); Entries", 100, 0.0, 0.05);
  trk_dzError_mu_hist = ibook.book1D("trk_dzError_mu", "Muon dz Error; dz Error (cm); Entries", 100, 0.0, 0.05);
  trk_qoverpError_mu_hist = ibook.book1D("trk_qoverpError_mu", "Muon q/p Error; q/p Error; Entries", 100, 0.0, 0.01);
  trk_lambdaError_mu_hist =
      ibook.book1D("trk_lambdaError_mu", "Muon Lambda Error; #lambda Error; Entries", 100, 0.0, 0.1);
  trk_phiError_mu_hist = ibook.book1D("trk_phiError_mu", "Muon Phi Error; #phi Error (rad); Entries", 100, 0.0, 0.01);
  trk_dsz_mu_hist = ibook.book1D("trk_dsz_mu", "Muon dsz; dsz (cm); Entries", 100, -2, 2);
  trk_dszError_mu_hist = ibook.book1D("trk_dszError_mu", "Muon dsz Error; dsz Error (cm); Entries", 100, 0.0, 0.05);
  trk_qoverp_lambda_cov_mu_hist =
      ibook.book1D("trk_qoverp_lambda_cov_mu", "Muon q/p-#lambda Covariance; Covariance; Entries", 100, -0.001, 0.001);
  trk_qoverp_phi_cov_mu_hist =
      ibook.book1D("trk_qoverp_phi_cov_mu", "Muon q/p-#phi Covariance; Covariance; Entries", 100, -0.001, 0.001);
  trk_qoverp_dxy_cov_mu_hist =
      ibook.book1D("trk_qoverp_dxy_cov_mu", "Muon q/p-dxy Covariance; Covariance; Entries", 100, -0.001, 0.001);
  trk_qoverp_dsz_cov_mu_hist =
      ibook.book1D("trk_qoverp_dsz_cov_mu", "Muon q/p-dsz Covariance; Covariance; Entries", 100, -0.001, 0.001);
  trk_lambda_phi_cov_mu_hist =
      ibook.book1D("trk_lambda_phi_cov_mu", "Muon Lambda-#phi Covariance; Covariance; Entries", 100, -0.001, 0.001);
  trk_lambda_dxy_cov_mu_hist =
      ibook.book1D("trk_lambda_dxy_cov_mu", "Muon Lambda-dxy Covariance; Covariance; Entries", 100, -0.001, 0.001);
  trk_lambda_dsz_cov_mu_hist =
      ibook.book1D("trk_lambda_dsz_cov_mu", "Muon Lambda-dsz Covariance; Covariance; Entries", 100, -0.001, 0.001);
  trk_phi_dxy_cov_mu_hist =
      ibook.book1D("trk_phi_dxy_cov_mu", "Muon Phi-dxy Covariance; Covariance; Entries", 100, -0.001, 0.001);
  trk_phi_dsz_cov_mu_hist =
      ibook.book1D("trk_phi_dsz_cov_mu", "Muon Phi-dsz Covariance; Covariance; Entries", 100, -0.001, 0.001);
  trk_dxy_dsz_cov_mu_hist =
      ibook.book1D("trk_dxy_dsz_cov_mu", "Muon dxy-dsz Covariance; Covariance; Entries", 100, -0.001, 0.001);
  trk_vx_mu_hist = ibook.book1D("trk_vx_mu", "Muon Tracker Vertex X; x (cm); Entries", 100, -0.5, 0.5);
  trk_vy_mu_hist = ibook.book1D("trk_vy_mu", "Muon Tracker Vertex Y; y (cm); Entries", 100, -0.5, 0.5);
  trk_vz_mu_hist = ibook.book1D("trk_vz_mu", "Muon Tracker Vertex Z; z (cm); Entries", 100, -20.0, 20.0);

  ibook.setCurrentFolder(outputInternalPath_ + "/PFJet");
  pt_pfj_hist = ibook.book1D("pt_pfj", "PF Jet pT; p_{T} (GeV); Entries", 100, 0.0, 150.0);
  eta_pfj_hist = ibook.book1D("eta_pfj", "PF Jet #eta; #eta; Entries", 100, -5.0, 5.0);
  phi_pfj_hist = ibook.book1D("phi_pfj", "PF Jet #phi; #phi (rad); Entries", 100, -3.14, 3.14);
  m_pfj_hist = ibook.book1D("m_pfj", "PF Jet Mass; Mass (GeV); Entries", 100, 0.0, 40.0);
  jetArea_pfj_hist = ibook.book1D("jetArea_pfj", "PF Jet Area; Area; Entries", 100, 0.0, 0.8);
  chargedHadronEnergy_pfj_hist =
      ibook.book1D("chargedHadronEnergy_pfj", "Charged Hadron Energy; Energy (GeV); Entries", 100, 0.0, 150.0);
  neutralHadronEnergy_pfj_hist =
      ibook.book1D("neutralHadronEnergy_pfj", "Neutral Hadron Energy; Energy (GeV); Entries", 100, 0.0, 600.0);
  photonEnergy_pfj_hist = ibook.book1D("photonEnergy_pfj", "Photon Energy; Energy (GeV); Entries", 100, 0.0, 90.0);
  electronEnergy_pfj_hist = ibook.book1D("electronEnergy_pfj", "Electron Energy; Energy (GeV); Entries", 100, 0.0, 3.0);
  muonEnergy_pfj_hist = ibook.book1D("muonEnergy_pfj", "Muon Energy; Energy (GeV); Entries", 100, 0.0, 3.0);
  HFHadronEnergy_pfj_hist =
      ibook.book1D("HFHadronEnergy_pfj", "HF Hadron Energy; Energy (GeV); Entries", 100, 0.0, 300.0);
  HFEMEnergy_pfj_hist = ibook.book1D("HFEMEnergy_pfj", "HF EM Energy; Energy (GeV); Entries", 100, 0.0, 300.0);
  chargedHadronMultiplicity_pfj_hist =
      ibook.book1D("chargedHadronMultiplicity_pfj", "Charged Hadron Multiplicity; Multiplicity; Entries", 50, 0, 25);
  neutralHadronMultiplicity_pfj_hist =
      ibook.book1D("neutralHadronMultiplicity_pfj", "Neutral Hadron Multiplicity; Multiplicity; Entries", 50, 0, 10);
  photonMultiplicity_pfj_hist =
      ibook.book1D("photonMultiplicity_pfj", "Photon Multiplicity; Multiplicity; Entries", 50, 0, 22);
  electronMultiplicity_pfj_hist =
      ibook.book1D("electronMultiplicity_pfj", "Electron Multiplicity; Multiplicity; Entries", 20, 0, 5);
  muonMultiplicity_pfj_hist =
      ibook.book1D("muonMultiplicity_pfj", "Muon Multiplicity; Multiplicity; Entries", 20, 0, 5);
  HFHadronMultiplicity_pfj_hist =
      ibook.book1D("HFHadronMultiplicity_pfj", "HF Hadron Multiplicity; Multiplicity; Entries", 20, 0, 20);
  HFEMMultiplicity_pfj_hist =
      ibook.book1D("HFEMMultiplicity_pfj", "HF EM Multiplicity; Multiplicity; Entries", 20, 0, 20);
  HOEnergy_pfj_hist = ibook.book1D("HOEnergy_pfj", "HO Energy; Energy (GeV); Entries", 100, 0.0, 5.0);
  csv_pfj_hist = ibook.book1D("csv_pfj", "Combined Secondary Vertex (CSV); CSV Score; Entries", 100, -0.5, 0.5);
  mvaDiscriminator_pfj_hist = ibook.book1D("mvaDiscriminator_pfj", "MVA Discriminator; Score; Entries", 100, -1.0, 1.0);

  ibook.setCurrentFolder(outputInternalPath_ + "/PrimaryVertex");
  x_pv_hist = ibook.book1D("x_pv", "Primary Vertex X Position; x (cm); Entries", 100, -0.5, 0.5);
  y_pv_hist = ibook.book1D("y_pv", "Primary Vertex Y Position; y (cm); Entries", 100, -0.5, 0.5);
  z_pv_hist = ibook.book1D("z_pv", "Primary Vertex Z Position; z (cm); Entries", 100, -20.0, 20.0);
  zError_pv_hist = ibook.book1D("zError_pv", "Primary Vertex Z Error; z Error (cm); Entries", 100, 0.0, 0.05);
  xError_pv_hist = ibook.book1D("xError_pv", "Primary Vertex X Error; x Error (cm); Entries", 100, 0.0, 0.05);
  yError_pv_hist = ibook.book1D("yError_pv", "Primary Vertex Y Error; y Error (cm); Entries", 100, 0.0, 0.05);
  tracksSize_pv_hist =
      ibook.book1D("tracksSize_pv", "Number of Tracks at Primary Vertex; Tracks; Entries", 100, 0, 100);
  chi2_pv_hist = ibook.book1D("chi2_pv", "Primary Vertex chi2; #chi^{2}; Entries", 100, 0.0, 50.0);
  ndof_pv_hist = ibook.book1D("ndof_pv", "Primary Vertex Ndof; Ndof; Entries", 100, 0, 100);
  isValidVtx_pv_hist = ibook.book1D("isValidVtx_pv", "Is Valid Primary Vertex?; 0 = False, 1 = True; Entries", 2, 0, 2);
  xyCov_pv_hist = ibook.book1D("xyCov_pv", "Primary Vertex XY Covariance; Cov(x,y); Entries", 100, -0.01, 0.01);
  xzCov_pv_hist = ibook.book1D("xzCov_pv", "Primary Vertex XZ Covariance; Cov(x,z); Entries", 100, -0.01, 0.01);
  yzCov_pv_hist = ibook.book1D("yzCov_pv", "Primary Vertex YZ Covariance; Cov(y,z); Entries", 100, -0.01, 0.01);

  ibook.setCurrentFolder(outputInternalPath_ + "/DisplacedVertex");
  x_vtx_hist = ibook.book1D("x_vtx", "Vertex X Position; x (cm); Entries", 100, -0.5, 0.5);
  y_vtx_hist = ibook.book1D("y_vtx", "Vertex Y Position; y (cm); Entries", 100, -0.5, 0.5);
  z_vtx_hist = ibook.book1D("z_vtx", "Vertex Z Position; z (cm); Entries", 100, -20.0, 20.0);
  zError_vtx_hist = ibook.book1D("zError_vtx", "Vertex Z Error; z Error (cm); Entries", 100, 0.0, 0.2);
  xError_vtx_hist = ibook.book1D("xError_vtx", "Vertex X Error; x Error (cm); Entries", 100, 0.0, 0.2);
  yError_vtx_hist = ibook.book1D("yError_vtx", "Vertex Y Error; y Error (cm); Entries", 100, 0.0, 0.2);
  tracksSize_vtx_hist = ibook.book1D("tracksSize_vtx", "Number of Tracks at Vertex; Tracks; Entries", 100, 0, 100);
  chi2_vtx_hist = ibook.book1D("chi2_vtx", "Vertex chi2; #chi^{2}; Entries", 100, 0.0, 5.0);
  ndof_vtx_hist = ibook.book1D("ndof_vtx", "Vertex Ndof; Ndof; Entries", 100, 0, 5);
  isValidVtx_vtx_hist = ibook.book1D("isValidVtx_vtx", "Is Valid Vertex?; 0 = False, 1 = True; Entries", 2, 0, 2);
  xyCov_vtx_hist = ibook.book1D("xyCov_vtx", "Vertex XY Covariance; Cov(x,y); Entries", 100, -0.01, 0.01);
  xzCov_vtx_hist = ibook.book1D("xzCov_vtx", "Vertex XZ Covariance; Cov(x,z); Entries", 100, -0.01, 0.01);
  yzCov_vtx_hist = ibook.book1D("yzCov_vtx", "Vertex YZ Covariance; Cov(y,z); Entries", 100, -0.01, 0.01);

  ibook.setCurrentFolder(outputInternalPath_ + "/Tracker");
  tk_pt_tk_hist = ibook.book1D("tk_pt_tk", "Tracker pT; p_{T} (GeV); Entries", 100, 0.0, 30.0);
  tk_eta_tk_hist = ibook.book1D("tk_eta_tk", "Tracker #eta; #eta; Entries", 100, -2.7, 2.7);
  tk_phi_tk_hist = ibook.book1D("tk_phi_tk", "Tracker #phi; #phi (rad); Entries", 100, -3.14, 3.14);
  tk_chi2_tk_hist = ibook.book1D("tk_chi2_tk", "Tracker chi2; #chi^{2}; Entries", 100, 0.0, 50.0);
  tk_ndof_tk_hist = ibook.book1D("tk_ndof_tk", "Tracker Ndof; Ndof; Entries", 100, 0, 50);
  tk_charge_tk_hist = ibook.book1D("tk_charge_tk", "Tracker Charge; Charge; Entries", 3, -1, 2);
  tk_dxy_tk_hist = ibook.book1D("tk_dxy_tk", "Tracker dxy; dxy (cm); Entries", 100, -0.5, 0.5);
  tk_dz_tk_hist = ibook.book1D("tk_dz_tk", "Tracker dz; dz (cm); Entries", 100, -20.0, 20.0);
  tk_nValidPixelHits_tk_hist = ibook.book1D("tk_nValidPixelHits_tk", "Valid Pixel Hits; Hits; Entries", 20, 0, 20);
  tk_nTrackerLayersWithMeasurement_tk_hist = ibook.book1D(
      "tk_nTrackerLayersWithMeasurement_tk", "Tracker Layers with Measurement; Layers; Entries", 20, 0, 20);
  tk_nValidStripHits_tk_hist = ibook.book1D("tk_nValidStripHits_tk", "Valid Strip Hits; Hits; Entries", 50, 0, 50);
  tk_qoverp_tk_hist = ibook.book1D("tk_qoverp_tk", "q/p; q/p; Entries", 100, -1.0, 1.0);
  tk_lambda_tk_hist = ibook.book1D("tk_lambda_tk", "Lambda; #lambda; Entries", 100, -2, 2);
  tk_dxy_Error_tk_hist = ibook.book1D("tk_dxy_Error_tk", "dxy Error; dxy Error (cm); Entries", 100, 0.0, 0.05);
  tk_dz_Error_tk_hist = ibook.book1D("tk_dz_Error_tk", "dz Error; dz Error (cm); Entries", 100, 0.0, 0.05);
  tk_qoverp_Error_tk_hist = ibook.book1D("tk_qoverp_Error_tk", "q/p Error; q/p Error; Entries", 100, 0.0, 0.05);
  tk_lambda_Error_tk_hist = ibook.book1D("tk_lambda_Error_tk", "Lambda Error; #lambda Error; Entries", 100, 0.0, 0.1);
  tk_phi_Error_tk_hist = ibook.book1D("tk_phi_Error_tk", "Phi Error; #phi Error (rad); Entries", 100, 0.0, 0.01);
  tk_dsz_tk_hist = ibook.book1D("tk_dsz_tk", "dsz; dsz (cm); Entries", 100, -2, 2);
  tk_dsz_Error_tk_hist = ibook.book1D("tk_dsz_Error_tk", "dsz Error; dsz Error (cm); Entries", 100, 0.0, 0.05);
  tk_vtxInd_tk_hist = ibook.book1D("tk_vtxInd_tk", "Vertex Index; Index; Entries", 50, 0, 50);
  tk_vx_tk_hist = ibook.book1D("tk_vx_tk", "Tracker Vertex X; x (cm); Entries", 100, -0.5, 0.5);
  tk_vy_tk_hist = ibook.book1D("tk_vy_tk", "Tracker Vertex Y; y (cm); Entries", 100, -0.5, 0.5);
  tk_vz_tk_hist = ibook.book1D("tk_vz_tk", "Tracker Vertex Z; z (cm); Entries", 100, -20.0, 20.0);
}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

void ScoutingCollectionMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("OutputInternalPath", "MY_FOLDER");
  desc.add<edm::InputTag>("triggerresults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("hltScoutingEgammaPacker"));
  desc.add<edm::InputTag>("muons", edm::InputTag("hltScoutingMuonPackerNoVtx"));
  desc.add<edm::InputTag>("pfcands", edm::InputTag("hltScoutingPFPacker"));
  desc.add<edm::InputTag>("photons", edm::InputTag("hltScoutingEgammaPacker"));
  desc.add<edm::InputTag>("pfjets", edm::InputTag("hltScoutingPFPacker"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("hltScoutingTrackPacker"));
  desc.add<edm::InputTag>("displacedVertices", edm::InputTag("hltScoutingMuonPackerNoVtx", "displacedVtx"));
  desc.add<edm::InputTag>("primaryVertices", edm::InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"));
  desc.add<edm::InputTag>("pfMetPt", edm::InputTag("hltScoutingPFPacker", "pfMetPt"));
  desc.add<edm::InputTag>("pfMetPhi", edm::InputTag("hltScoutingPFPacker", "pfMetPhi"));
  desc.add<edm::InputTag>("rho", edm::InputTag("hltScoutingPFPacker", "rho"));
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ScoutingCollectionMonitor);
