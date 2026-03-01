// -*- C++ -*-
//
// Package:    DQM/HLTEvF
// Class:      ScoutingCollectionMonitor
//
/**\class ScoutingCollectionMonitor ScoutingCollectionMonitor.cc 
          DQM/HLTEvF/plugins/ScoutingCollectionMonitor.cc

Description: ScoutingCollectionMonitor is developed to enable monitoring of several scouting objects and comparisons for the NGT demonstrator
It is based on the preexisting work of the scouting group and can be found at git@github.com:CMS-Run3ScoutingTools/Run3ScoutingAnalysisTools.git

*/
//
// Original Author:  Jessica Prendi
//         Created:  Thu, 17 Apr 2025 14:15:08 GMT
//
//

// system include files
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include <TLorentzVector.h>

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/Scouting/interface/Run3ScoutingEBRecHit.h"
#include "DataFormats/Scouting/interface/Run3ScoutingEERecHit.h"
#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
#include "DataFormats/Scouting/interface/Run3ScoutingHBHERecHit.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"
#include "DataFormats/Scouting/interface/Run3ScoutingParticle.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"
#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"

//
// class declaration
//

class ScoutingCollectionMonitor : public DQMEDAnalyzer {
public:
  explicit ScoutingCollectionMonitor(const edm::ParameterSet&);
  ~ScoutingCollectionMonitor() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  template <typename T>
  void setToken(edm::EDGetTokenT<T>& token, const edm::ParameterSet& iConfig, std::string name) {
    const auto inputTag = iConfig.getParameter<edm::InputTag>(name);
    if (!inputTag.encode().empty()) {
      token = mayConsume<T>(inputTag);
    }
  }

  template <typename T>
  bool getValidHandle(const edm::Event& iEvent,
                      const edm::EDGetTokenT<T>& token,
                      edm::Handle<T>& handle,
                      const std::string& label);

  static inline std::pair<float, float> trk_vtx_offSet(const Run3ScoutingTrack& tk, const Run3ScoutingVertex& vtx) {
    const auto pt = tk.tk_pt();
    const auto phi = tk.tk_phi();
    const auto eta = tk.tk_eta();

    const auto px = pt * std::cos(phi);
    const auto py = pt * std::sin(phi);
    const auto pz = pt * std::sinh(eta);
    const auto pt2 = pt * pt;

    const auto dx = tk.tk_vx() - vtx.x();
    const auto dy = tk.tk_vy() - vtx.y();
    const auto dz = tk.tk_vz() - vtx.z();

    const auto tk_dxyPV = (-dx * py + dy * px) / pt;
    const auto tk_dzPV = dz - (dx * px + dy * py) * pz / pt2;

    return {tk_dxyPV, tk_dzPV};
  }

  const bool onlyScouting_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> muonsToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> muonsVtxToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingElectron>> electronsToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> primaryVerticesToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> verticesToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> verticesNoVtxToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingPhoton>> photonsToken_;
  const edm::EDGetTokenT<double> rhoToken_;
  const edm::EDGetTokenT<double> pfMetPhiToken_;
  const edm::EDGetTokenT<double> pfMetPtToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingParticle>> pfcandsToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingPFJet>> pfjetsToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingTrack>> tracksToken_;
  const edm::EDGetTokenT<OnlineLuminosityRecord> onlineMetaDataDigisToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const std::string topfoldername_;

  // calo rechits (only 2025 V1.3 onwards, see https://its.cern.ch/jira/browse/CMSHLT-3607)
  edm::EDGetTokenT<Run3ScoutingEBRecHitCollection> ebRecHitsToken_;
  edm::EDGetTokenT<Run3ScoutingEERecHitCollection> eeRecHitsToken_;
  edm::EDGetTokenT<Run3ScoutingEBRecHitCollection> ebCleanedRecHitsToken_;
  edm::EDGetTokenT<Run3ScoutingEERecHitCollection> eeCleanedRecHitsToken_;
  edm::EDGetTokenT<Run3ScoutingHBHERecHitCollection> hbheRecHitsToken_;

  // Multiplicity histograms
  dqm::reco::MonitorElement* nTracks_hist;
  dqm::reco::MonitorElement* nPrimaryVertices_hist;
  dqm::reco::MonitorElement* nDisplacedVertices_hist;
  dqm::reco::MonitorElement* nDisplacedVerticesNoVtx_hist;
  dqm::reco::MonitorElement* nMuons_hist;
  dqm::reco::MonitorElement* nMuonsVtx_hist;
  dqm::reco::MonitorElement* nElectrons_hist;
  dqm::reco::MonitorElement* nPhotons_hist;
  dqm::reco::MonitorElement* nPFJets_hist;
  dqm::reco::MonitorElement* nPFCands_hist;

  // pv vs PU and rho vs PU plots
  int primaryVertex_counter = 0;
  float avgPileUp;
  dqm::reco::MonitorElement* PVvsPU_hist;
  dqm::reco::MonitorElement* rhovsPU_hist;

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

  // the following variables make sense only if there is a Track

  dqm::reco::MonitorElement* PF_normchi2_211_hist;
  dqm::reco::MonitorElement* PF_normchi2_n211_hist;
  dqm::reco::MonitorElement* PF_normchi2_13_hist;
  dqm::reco::MonitorElement* PF_normchi2_n13_hist;

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
  dqm::reco::MonitorElement* seedId_pho_hist;
  dqm::reco::MonitorElement* nClusters_pho_hist;
  dqm::reco::MonitorElement* nCrystals_pho_hist;
  dqm::reco::MonitorElement* rechitZeroSuppression_pho_hist;

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

  // muon histograms (index 0: noVtx, index1: Vtx)
  dqm::reco::MonitorElement* pt_mu_hist[2];
  dqm::reco::MonitorElement* eta_mu_hist[2];
  dqm::reco::MonitorElement* phi_mu_hist[2];
  dqm::reco::MonitorElement* type_mu_hist[2];
  dqm::reco::MonitorElement* charge_mu_hist[2];
  dqm::reco::MonitorElement* normalizedChi2_mu_hist[2];
  dqm::reco::MonitorElement* ecalIso_mu_hist[2];
  dqm::reco::MonitorElement* hcalIso_mu_hist[2];
  dqm::reco::MonitorElement* trackIso_mu_hist[2];
  dqm::reco::MonitorElement* nValidStandAloneMuonHits_mu_hist[2];
  dqm::reco::MonitorElement* nStandAloneMuonMatchedStations_mu_hist[2];
  dqm::reco::MonitorElement* nValidRecoMuonHits_mu_hist[2];
  dqm::reco::MonitorElement* nRecoMuonChambers_mu_hist[2];
  dqm::reco::MonitorElement* nRecoMuonChambersCSCorDT_mu_hist[2];
  dqm::reco::MonitorElement* nRecoMuonMatches_mu_hist[2];
  dqm::reco::MonitorElement* nRecoMuonMatchedStations_mu_hist[2];
  dqm::reco::MonitorElement* nRecoMuonExpectedMatchedStations_mu_hist[2];
  dqm::reco::MonitorElement* recoMuonStationMask_mu_hist[2];
  dqm::reco::MonitorElement* nRecoMuonMatchedRPCLayers_mu_hist[2];
  dqm::reco::MonitorElement* recoMuonRPClayerMask_mu_hist[2];
  dqm::reco::MonitorElement* nValidPixelHits_mu_hist[2];
  dqm::reco::MonitorElement* nValidStripHits_mu_hist[2];
  dqm::reco::MonitorElement* nPixelLayersWithMeasurement_mu_hist[2];
  dqm::reco::MonitorElement* nTrackerLayersWithMeasurement_mu_hist[2];
  dqm::reco::MonitorElement* trk_chi2_mu_hist[2];
  dqm::reco::MonitorElement* trk_ndof_mu_hist[2];
  dqm::reco::MonitorElement* trk_dxy_mu_hist[2];
  dqm::reco::MonitorElement* trk_dz_mu_hist[2];
  dqm::reco::MonitorElement* trk_qoverp_mu_hist[2];
  dqm::reco::MonitorElement* trk_lambda_mu_hist[2];
  dqm::reco::MonitorElement* trk_pt_mu_hist[2];
  dqm::reco::MonitorElement* trk_phi_mu_hist[2];
  dqm::reco::MonitorElement* trk_eta_mu_hist[2];
  dqm::reco::MonitorElement* trk_dxyError_mu_hist[2];
  dqm::reco::MonitorElement* trk_dzError_mu_hist[2];
  dqm::reco::MonitorElement* trk_qoverpError_mu_hist[2];
  dqm::reco::MonitorElement* trk_lambdaError_mu_hist[2];
  dqm::reco::MonitorElement* trk_phiError_mu_hist[2];
  dqm::reco::MonitorElement* trk_dsz_mu_hist[2];
  dqm::reco::MonitorElement* trk_dszError_mu_hist[2];
  dqm::reco::MonitorElement* trk_qoverp_lambda_cov_mu_hist[2];
  dqm::reco::MonitorElement* trk_qoverp_phi_cov_mu_hist[2];
  dqm::reco::MonitorElement* trk_qoverp_dxy_cov_mu_hist[2];
  dqm::reco::MonitorElement* trk_qoverp_dsz_cov_mu_hist[2];
  dqm::reco::MonitorElement* trk_lambda_phi_cov_mu_hist[2];
  dqm::reco::MonitorElement* trk_lambda_dxy_cov_mu_hist[2];
  dqm::reco::MonitorElement* trk_lambda_dsz_cov_mu_hist[2];
  dqm::reco::MonitorElement* trk_phi_dxy_cov_mu_hist[2];
  dqm::reco::MonitorElement* trk_phi_dsz_cov_mu_hist[2];
  dqm::reco::MonitorElement* trk_dxy_dsz_cov_mu_hist[2];
  dqm::reco::MonitorElement* trk_vx_mu_hist[2];
  dqm::reco::MonitorElement* trk_vy_mu_hist[2];
  dqm::reco::MonitorElement* trk_vz_mu_hist[2];

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

  // displaced vertex histograms (index 0: Vtx, index1: NoVtx)
  dqm::reco::MonitorElement* x_vtx_hist[2];
  dqm::reco::MonitorElement* y_vtx_hist[2];
  dqm::reco::MonitorElement* z_vtx_hist[2];
  dqm::reco::MonitorElement* zError_vtx_hist[2];
  dqm::reco::MonitorElement* xError_vtx_hist[2];
  dqm::reco::MonitorElement* yError_vtx_hist[2];
  dqm::reco::MonitorElement* tracksSize_vtx_hist[2];
  dqm::reco::MonitorElement* chi2_vtx_hist[2];
  dqm::reco::MonitorElement* ndof_vtx_hist[2];
  dqm::reco::MonitorElement* isValidVtx_vtx_hist[2];
  dqm::reco::MonitorElement* xyCov_vtx_hist[2];
  dqm::reco::MonitorElement* xzCov_vtx_hist[2];
  dqm::reco::MonitorElement* yzCov_vtx_hist[2];

  // general tracking histograms
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
  dqm::reco::MonitorElement* tk_chi2_ndof_tk_hist;
  dqm::reco::MonitorElement* tk_chi2_prob_hist;
  dqm::reco::MonitorElement* tk_PV_dxy_hist;
  dqm::reco::MonitorElement* tk_PV_dz_hist;
  dqm::reco::MonitorElement* tk_BS_dxy_hist;
  dqm::reco::MonitorElement* tk_BS_dz_hist;

  // calo rechits histrograms (ECAL has two version, cleaned and unclean)
  dqm::reco::MonitorElement* ebRecHitsNumber_hist[2];
  dqm::reco::MonitorElement* ebRecHits_energy_hist[2];
  dqm::reco::MonitorElement* ebRecHits_time_hist[2];
  dqm::reco::MonitorElement* ebRecHitsEtaPhiMap[2];
  dqm::reco::MonitorElement* eeRecHitsNumber_hist[2];
  dqm::reco::MonitorElement* eeRecHits_energy_hist[2];
  dqm::reco::MonitorElement* eeRecHits_time_hist[2];
  dqm::reco::MonitorElement* eePlusRecHitsXYMap[2];
  dqm::reco::MonitorElement* eeMinusRecHitsXYMap[2];

  // three MEs (HBHE, HB, HE)
  dqm::reco::MonitorElement* hbheRecHitsNumber_hist[3];
  dqm::reco::MonitorElement* hbheRecHits_energy_hist[3];
  dqm::reco::MonitorElement* hbheRecHits_time_hist[3];
  dqm::reco::MonitorElement* hbheRecHits_energy_egt5_hist[3];
  dqm::reco::MonitorElement* hbheRecHits_time_egt5_hist[3];

  // separate maps for each subdetector
  dqm::reco::MonitorElement* hbheRecHitsEtaPhiMap;
  dqm::reco::MonitorElement* hbRecHitsEtaPhiMap;
  dqm::reco::MonitorElement* heRecHitsEtaPhiMap;
};

//
// constructors and destructor
//
ScoutingCollectionMonitor::ScoutingCollectionMonitor(const edm::ParameterSet& iConfig)
    : onlyScouting_(iConfig.getParameter<bool>("onlyScouting")),
      muonsToken_(consumes<std::vector<Run3ScoutingMuon>>(iConfig.getParameter<edm::InputTag>("muons"))),
      muonsVtxToken_(consumes<std::vector<Run3ScoutingMuon>>(iConfig.getParameter<edm::InputTag>("muonsVtx"))),
      electronsToken_(consumes<std::vector<Run3ScoutingElectron>>(iConfig.getParameter<edm::InputTag>("electrons"))),
      primaryVerticesToken_(
          consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("primaryVertices"))),
      verticesToken_(
          consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("displacedVertices"))),
      verticesNoVtxToken_(
          consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("displacedVerticesNoVtx"))),
      photonsToken_(consumes<std::vector<Run3ScoutingPhoton>>(iConfig.getParameter<edm::InputTag>("photons"))),
      rhoToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("rho"))),
      pfMetPhiToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("pfMetPhi"))),
      pfMetPtToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("pfMetPt"))),
      pfcandsToken_(consumes<std::vector<Run3ScoutingParticle>>(iConfig.getParameter<edm::InputTag>("pfcands"))),
      pfjetsToken_(consumes<std::vector<Run3ScoutingPFJet>>(iConfig.getParameter<edm::InputTag>("pfjets"))),
      tracksToken_(consumes<std::vector<Run3ScoutingTrack>>(iConfig.getParameter<edm::InputTag>("tracks"))),
      onlineMetaDataDigisToken_(consumes(iConfig.getParameter<edm::InputTag>("onlineMetaDataDigis"))),
      beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      topfoldername_(iConfig.getParameter<std::string>("topfoldername")) {
  setToken(ebRecHitsToken_, iConfig, "pfRecHitsEB");
  setToken(eeRecHitsToken_, iConfig, "pfRecHitsEE");
  setToken(ebCleanedRecHitsToken_, iConfig, "pfCleanedRecHitsEB");
  setToken(eeCleanedRecHitsToken_, iConfig, "pfCleanedRecHitsEE");
  setToken(hbheRecHitsToken_, iConfig, "pfRecHitsHBHE");
}

//
// member functions
//
template <typename T>
bool ScoutingCollectionMonitor::getValidHandle(const edm::Event& iEvent,
                                               const edm::EDGetTokenT<T>& token,
                                               edm::Handle<T>& handle,
                                               const std::string& label) {
  iEvent.getByToken(token, handle);
  if (!handle.isValid()) {
    edm::LogWarning("ScoutingAnalyzer") << "Invalid handle for " << label;
    return false;
  }
  return true;
}

// ------------ method called for each event  ------------
void ScoutingCollectionMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using namespace reco;

  // all the handles needed
  edm::Handle<double> rhoH;
  edm::Handle<double> pfMetPhiH;
  edm::Handle<double> pfMetPtH;
  edm::Handle<std::vector<Run3ScoutingParticle>> pfcandsH;
  edm::Handle<std::vector<Run3ScoutingPhoton>> photonsH;
  edm::Handle<std::vector<Run3ScoutingElectron>> electronsH;
  edm::Handle<std::vector<Run3ScoutingMuon>> muonsH;
  edm::Handle<std::vector<Run3ScoutingMuon>> muonsVtxH;
  edm::Handle<std::vector<Run3ScoutingPFJet>> PFjetsH;
  edm::Handle<std::vector<Run3ScoutingVertex>> verticesH;
  edm::Handle<std::vector<Run3ScoutingVertex>> verticesNoVtxH;
  edm::Handle<std::vector<Run3ScoutingVertex>> primaryVerticesH;
  edm::Handle<std::vector<Run3ScoutingTrack>> tracksH;
  edm::Handle<OnlineLuminosityRecord> onlineMetaDataDigisHandle;

  if (!getValidHandle(iEvent, rhoToken_, rhoH, "rho") ||
      !getValidHandle(iEvent, pfMetPhiToken_, pfMetPhiH, "MET phi") ||
      !getValidHandle(iEvent, pfMetPtToken_, pfMetPtH, "MET pT") ||
      !getValidHandle(iEvent, pfcandsToken_, pfcandsH, "PF candidates") ||
      !getValidHandle(iEvent, photonsToken_, photonsH, "photons") ||
      !getValidHandle(iEvent, electronsToken_, electronsH, "electrons") ||
      !getValidHandle(iEvent, muonsToken_, muonsH, "muons") ||
      !getValidHandle(iEvent, muonsVtxToken_, muonsVtxH, "muonsVtx") ||
      !getValidHandle(iEvent, pfjetsToken_, PFjetsH, "PF jets") ||
      !getValidHandle(iEvent, verticesToken_, verticesH, "vertices") ||
      !getValidHandle(iEvent, verticesNoVtxToken_, verticesNoVtxH, "verticesNoVtx") ||
      !getValidHandle(iEvent, primaryVerticesToken_, primaryVerticesH, "primary vertices") ||
      !getValidHandle(iEvent, tracksToken_, tracksH, "tracks")) {
    return;
  }

  // get pile up
  if (!onlyScouting_) {
    if (!getValidHandle(iEvent, onlineMetaDataDigisToken_, onlineMetaDataDigisHandle, "avgPileUp")) {
      return;
    }
    avgPileUp = onlineMetaDataDigisHandle->avgPileUp();
    rhovsPU_hist->Fill(avgPileUp, *rhoH);
  }

  // put stuff in histogram
  rho_hist->Fill(*rhoH);
  pfMetPhi_hist->Fill(*pfMetPhiH);
  pfMetPt_hist->Fill(*pfMetPtH);

  // --- Fill multiplicity histograms ---
  nTracks_hist->Fill(tracksH->size());
  nPrimaryVertices_hist->Fill(primaryVerticesH->size());
  nDisplacedVertices_hist->Fill(verticesH->size());
  nDisplacedVerticesNoVtx_hist->Fill(verticesNoVtxH->size());
  nMuons_hist->Fill(muonsH->size());
  nMuonsVtx_hist->Fill(muonsVtxH->size());
  nElectrons_hist->Fill(electronsH->size());
  nPhotons_hist->Fill(photonsH->size());
  nPFJets_hist->Fill(PFjetsH->size());
  nPFCands_hist->Fill(pfcandsH->size());

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
        break;

      case 22:
        PF_pT_22_hist->Fill(cand.pt());
        PF_eta_22_hist->Fill(cand.eta());
        PF_phi_22_hist->Fill(cand.phi());
        PF_vertex_22_hist->Fill(cand.vertex());
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
        break;

      case 2:
        PF_pT_2_hist->Fill(cand.pt());
        PF_eta_2_hist->Fill(cand.eta());
        PF_phi_2_hist->Fill(cand.phi());
        PF_vertex_2_hist->Fill(cand.vertex());
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

  // Apply to both collections
  auto fillMuonHistograms = [&](const auto& mu, size_t idx) {
    pt_mu_hist[idx]->Fill(mu.pt());
    eta_mu_hist[idx]->Fill(mu.eta());
    phi_mu_hist[idx]->Fill(mu.phi());
    type_mu_hist[idx]->Fill(mu.type());
    charge_mu_hist[idx]->Fill(mu.charge());
    normalizedChi2_mu_hist[idx]->Fill(mu.normalizedChi2());
    ecalIso_mu_hist[idx]->Fill(mu.ecalIso());
    hcalIso_mu_hist[idx]->Fill(mu.hcalIso());
    trackIso_mu_hist[idx]->Fill(mu.trackIso());
    nValidStandAloneMuonHits_mu_hist[idx]->Fill(mu.nValidStandAloneMuonHits());
    nStandAloneMuonMatchedStations_mu_hist[idx]->Fill(mu.nStandAloneMuonMatchedStations());
    nValidRecoMuonHits_mu_hist[idx]->Fill(mu.nValidRecoMuonHits());
    nRecoMuonChambers_mu_hist[idx]->Fill(mu.nRecoMuonChambers());
    nRecoMuonChambersCSCorDT_mu_hist[idx]->Fill(mu.nRecoMuonChambersCSCorDT());
    nRecoMuonMatches_mu_hist[idx]->Fill(mu.nRecoMuonMatches());
    nRecoMuonMatchedStations_mu_hist[idx]->Fill(mu.nRecoMuonMatchedStations());
    nRecoMuonExpectedMatchedStations_mu_hist[idx]->Fill(mu.nRecoMuonExpectedMatchedStations());
    recoMuonStationMask_mu_hist[idx]->Fill(mu.recoMuonStationMask());
    nRecoMuonMatchedRPCLayers_mu_hist[idx]->Fill(mu.nRecoMuonMatchedRPCLayers());
    recoMuonRPClayerMask_mu_hist[idx]->Fill(mu.recoMuonRPClayerMask());
    nValidPixelHits_mu_hist[idx]->Fill(mu.nValidPixelHits());
    nValidStripHits_mu_hist[idx]->Fill(mu.nValidStripHits());
    nPixelLayersWithMeasurement_mu_hist[idx]->Fill(mu.nPixelLayersWithMeasurement());
    nTrackerLayersWithMeasurement_mu_hist[idx]->Fill(mu.nTrackerLayersWithMeasurement());
    trk_chi2_mu_hist[idx]->Fill(mu.trk_chi2());
    trk_ndof_mu_hist[idx]->Fill(mu.trk_ndof());
    trk_dxy_mu_hist[idx]->Fill(mu.trk_dxy());
    trk_dz_mu_hist[idx]->Fill(mu.trk_dz());
    trk_qoverp_mu_hist[idx]->Fill(mu.trk_qoverp());
    trk_lambda_mu_hist[idx]->Fill(mu.trk_lambda());
    trk_pt_mu_hist[idx]->Fill(mu.trk_pt());
    trk_phi_mu_hist[idx]->Fill(mu.trk_phi());
    trk_eta_mu_hist[idx]->Fill(mu.trk_eta());
    trk_dxyError_mu_hist[idx]->Fill(mu.trk_dxyError());
    trk_dzError_mu_hist[idx]->Fill(mu.trk_dzError());
    trk_qoverpError_mu_hist[idx]->Fill(mu.trk_qoverpError());
    trk_lambdaError_mu_hist[idx]->Fill(mu.trk_lambdaError());
    trk_phiError_mu_hist[idx]->Fill(mu.trk_phiError());
    trk_dsz_mu_hist[idx]->Fill(mu.trk_dsz());
    trk_dszError_mu_hist[idx]->Fill(mu.trk_dszError());
    trk_qoverp_lambda_cov_mu_hist[idx]->Fill(mu.trk_qoverp_lambda_cov());
    trk_qoverp_phi_cov_mu_hist[idx]->Fill(mu.trk_qoverp_phi_cov());
    trk_qoverp_dxy_cov_mu_hist[idx]->Fill(mu.trk_qoverp_dxy_cov());
    trk_qoverp_dsz_cov_mu_hist[idx]->Fill(mu.trk_qoverp_dsz_cov());
    trk_lambda_phi_cov_mu_hist[idx]->Fill(mu.trk_lambda_phi_cov());
    trk_lambda_dxy_cov_mu_hist[idx]->Fill(mu.trk_lambda_dxy_cov());
    trk_lambda_dsz_cov_mu_hist[idx]->Fill(mu.trk_lambda_dsz_cov());
    trk_phi_dxy_cov_mu_hist[idx]->Fill(mu.trk_phi_dxy_cov());
    trk_phi_dsz_cov_mu_hist[idx]->Fill(mu.trk_phi_dsz_cov());
    trk_dxy_dsz_cov_mu_hist[idx]->Fill(mu.trk_dxy_dsz_cov());
    trk_vx_mu_hist[idx]->Fill(mu.trk_vx());
    trk_vy_mu_hist[idx]->Fill(mu.trk_vy());
    trk_vz_mu_hist[idx]->Fill(mu.trk_vz());
  };

  // muon histograms (index 0: noVtx)
  for (const auto& mu : *muonsH)
    fillMuonHistograms(mu, 0);

  // muon histograms (index1: Vtx)
  for (const auto& mu : *muonsVtxH)
    fillMuonHistograms(mu, 1);

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
    mvaDiscriminator_pfj_hist->Fill(jet.mvaDiscriminator());
  }

  primaryVertex_counter = 0;
  // fill all the primary vertices histograms
  for (const auto& vtx : *primaryVerticesH) {
    primaryVertex_counter++;
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

  if (!onlyScouting_) {
    PVvsPU_hist->Fill(avgPileUp, primaryVertex_counter);
  }

  // fill all the displaced vertices histograms
  auto fillVtxHistograms = [&](const auto& vtx, size_t idx) {
    x_vtx_hist[idx]->Fill(vtx.x());
    y_vtx_hist[idx]->Fill(vtx.y());
    z_vtx_hist[idx]->Fill(vtx.z());
    zError_vtx_hist[idx]->Fill(vtx.zError());
    xError_vtx_hist[idx]->Fill(vtx.xError());
    yError_vtx_hist[idx]->Fill(vtx.yError());
    tracksSize_vtx_hist[idx]->Fill(vtx.tracksSize());
    chi2_vtx_hist[idx]->Fill(vtx.chi2());
    ndof_vtx_hist[idx]->Fill(vtx.ndof());
    isValidVtx_vtx_hist[idx]->Fill(vtx.isValidVtx());
    xyCov_vtx_hist[idx]->Fill(vtx.xyCov());
    xzCov_vtx_hist[idx]->Fill(vtx.xzCov());
    yzCov_vtx_hist[idx]->Fill(vtx.yzCov());
  };

  // displaced vertex histograms with MuonVtx (index 0: Vtx)
  for (const auto& vtx : *verticesH)
    fillVtxHistograms(vtx, 0);

  // displaced vertex histograms with MuonNoVtx (index1: NoVtx)
  for (const auto& vtx : *verticesNoVtxH)
    fillVtxHistograms(vtx, 1);

  // determine the beamspot position (if it exists in the event)
  std::unique_ptr<Run3ScoutingVertex> beamspotVertex{nullptr};
  edm::Handle<reco::BeamSpot> beamSpotH;
  if (getValidHandle(iEvent, beamSpotToken_, beamSpotH, "beamSpot")) {
    const auto& beamspot = *beamSpotH;
    beamspotVertex = std::make_unique<Run3ScoutingVertex>(
        beamspot.x0(), beamspot.y0(), beamspot.z0(), 0., 0., 0., 0., 0., true, 0., 0., 0., 0);
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
    tk_dsz_tk_hist->Fill(tk.tk_dsz());
    tk_dsz_Error_tk_hist->Fill(tk.tk_dsz_Error());
    tk_vtxInd_tk_hist->Fill(tk.tk_vtxInd());
    tk_vx_tk_hist->Fill(tk.tk_vx());
    tk_vy_tk_hist->Fill(tk.tk_vy());
    tk_vz_tk_hist->Fill(tk.tk_vz());
    tk_chi2_ndof_tk_hist->Fill(tk.tk_chi2() / tk.tk_ndof());
    tk_chi2_prob_hist->Fill(TMath::Prob(tk.tk_chi2(), tk.tk_ndof()));

    // initialize the impact parameters to large values
    std::pair<float, float> best_offset{9999.f, 99999.f};

    // loop on all the vertices and find the closest one
    for (const auto& vtx : *primaryVerticesH) {
      const auto offset = trk_vtx_offSet(tk, vtx);
      if (std::abs(offset.second) < std::abs(best_offset.second)) {
        best_offset = offset;
      }
    }

    tk_PV_dxy_hist->Fill(best_offset.first);
    tk_PV_dz_hist->Fill(best_offset.second);

    // skip beamspot-based plots if not valid
    if (beamspotVertex) {
      auto bs_offset = trk_vtx_offSet(tk, *beamspotVertex);
      tk_BS_dxy_hist->Fill(bs_offset.first);
      tk_BS_dz_hist->Fill(bs_offset.second);
    }
  }

  // Define helper lambdas for EB and EE rechits
  auto fillEBHistograms = [](const auto& rechits,
                             int index,
                             dqm::reco::MonitorElement* numberHist[2],
                             dqm::reco::MonitorElement* etaPhiMap[2],
                             dqm::reco::MonitorElement* energyHist[2],
                             dqm::reco::MonitorElement* timeHist[2]) {
    numberHist[index]->Fill(rechits.size());
    for (const auto& hit : rechits) {
      EBDetId id(hit.detId());
      etaPhiMap[index]->Fill(id.ieta(), id.iphi());
      energyHist[index]->Fill(hit.energy());
      timeHist[index]->Fill(hit.time());
    }
  };

  auto fillEEHistograms = [](const auto& rechits,
                             int index,
                             dqm::reco::MonitorElement* numberHist[2],
                             dqm::reco::MonitorElement* plusXYMap[2],
                             dqm::reco::MonitorElement* minusXYMap[2],
                             dqm::reco::MonitorElement* energyHist[2],
                             dqm::reco::MonitorElement* timeHist[2]) {
    numberHist[index]->Fill(rechits.size());
    for (const auto& hit : rechits) {
      EEDetId id(hit.detId());
      if (id.zside() > 0) {
        plusXYMap[index]->Fill(id.ix(), id.iy());
      } else {
        minusXYMap[index]->Fill(id.ix(), id.iy());
      }
      energyHist[index]->Fill(hit.energy());
      timeHist[index]->Fill(hit.time());
    }
  };

  // Process uncleaned EB rechits
  edm::Handle<Run3ScoutingEBRecHitCollection> ebRecHitsH;
  if (!ebRecHitsToken_.isUninitialized() && getValidHandle(iEvent, ebRecHitsToken_, ebRecHitsH, "pfRecHitsEB")) {
    fillEBHistograms(
        *ebRecHitsH, 0, ebRecHitsNumber_hist, ebRecHitsEtaPhiMap, ebRecHits_energy_hist, ebRecHits_time_hist);
  }

  // Process uncleaned EE rechits
  edm::Handle<Run3ScoutingEERecHitCollection> eeRecHitsH;
  if (!eeRecHitsToken_.isUninitialized() && getValidHandle(iEvent, eeRecHitsToken_, eeRecHitsH, "pfRecHitsEE")) {
    fillEEHistograms(*eeRecHitsH,
                     0,
                     eeRecHitsNumber_hist,
                     eePlusRecHitsXYMap,
                     eeMinusRecHitsXYMap,
                     eeRecHits_energy_hist,
                     eeRecHits_time_hist);
  }

  // Process cleaned EB rechits
  edm::Handle<Run3ScoutingEBRecHitCollection> ebRecHitsCleanedH;
  if (!ebCleanedRecHitsToken_.isUninitialized() &&
      getValidHandle(iEvent, ebCleanedRecHitsToken_, ebRecHitsCleanedH, "pfCleanedRecHitsEB")) {
    fillEBHistograms(
        *ebRecHitsCleanedH, 1, ebRecHitsNumber_hist, ebRecHitsEtaPhiMap, ebRecHits_energy_hist, ebRecHits_time_hist);
  }

  // Process cleaned EE rechits
  edm::Handle<Run3ScoutingEERecHitCollection> eeRecHitsCleanedH;
  if (!eeCleanedRecHitsToken_.isUninitialized() &&
      getValidHandle(iEvent, eeCleanedRecHitsToken_, eeRecHitsCleanedH, "pfCleanedRecHitsEE")) {
    fillEEHistograms(*eeRecHitsCleanedH,
                     1,
                     eeRecHitsNumber_hist,
                     eePlusRecHitsXYMap,
                     eeMinusRecHitsXYMap,
                     eeRecHits_energy_hist,
                     eeRecHits_time_hist);
  }

  // counter of rechits
  size_t nHBRechits{0};
  size_t nHERechits{0};

  // process the HBHE rechits
  edm::Handle<Run3ScoutingHBHERecHitCollection> hbheRecHitsH;
  if (!hbheRecHitsToken_.isUninitialized() &&
      getValidHandle(iEvent, hbheRecHitsToken_, hbheRecHitsH, "pfRecHitsHBHE")) {
    hbheRecHitsNumber_hist[0]->Fill(hbheRecHitsH->size());
    for (const auto& hbheRecHit : *hbheRecHitsH) {
      const bool isStiffRecHit = (hbheRecHit.energy() > 5);

      hbheRecHits_energy_hist[0]->Fill(hbheRecHit.energy());
      hbheRecHits_time_hist[0]->Fill(hbheRecHit.time());
      if (isStiffRecHit) {
        hbheRecHits_energy_egt5_hist[0]->Fill(hbheRecHit.energy());
        hbheRecHits_time_egt5_hist[0]->Fill(hbheRecHit.time());
      }

      HcalDetId hcalid(hbheRecHit.detId());
      hbheRecHitsEtaPhiMap->Fill(hcalid.ieta(), hcalid.iphi());
      const auto& subdet = hcalid.subdetId();
      if (subdet == 1) {  // HB
        nHBRechits++;
        hbRecHitsEtaPhiMap->Fill(hcalid.ieta(), hcalid.iphi());
        hbheRecHits_energy_hist[1]->Fill(hbheRecHit.energy());
        hbheRecHits_time_hist[1]->Fill(hbheRecHit.time());
        if (isStiffRecHit) {
          hbheRecHits_energy_egt5_hist[1]->Fill(hbheRecHit.energy());
          hbheRecHits_time_egt5_hist[1]->Fill(hbheRecHit.time());
        }
      } else {  // HE
        nHERechits++;
        heRecHitsEtaPhiMap->Fill(hcalid.ieta(), hcalid.iphi());
        hbheRecHits_energy_hist[2]->Fill(hbheRecHit.energy());
        hbheRecHits_time_hist[2]->Fill(hbheRecHit.time());
        if (isStiffRecHit) {
          hbheRecHits_energy_egt5_hist[2]->Fill(hbheRecHit.energy());
          hbheRecHits_time_egt5_hist[2]->Fill(hbheRecHit.time());
        }
      }
    }
    // check that rechits size is the same
    assert(hbheRecHitsH->size() == (nHBRechits + nHERechits));
    hbheRecHitsNumber_hist[1]->Fill(nHBRechits);
    hbheRecHitsNumber_hist[2]->Fill(nHERechits);
  }
}

// ------------ method called once each job just before starting event loop  ------------
void ScoutingCollectionMonitor::bookHistograms(DQMStore::IBooker& ibook,
                                               edm::Run const& run,
                                               edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(topfoldername_);

  // Book multiplicity histograms in the topfolder
  nTracks_hist = ibook.book1D("nTracks", "Number of Tracks;N_{tracks};Entries", 400, 0, 400);
  nPrimaryVertices_hist = ibook.book1D("nPrimaryVertices", "Number of Primary Vertices;N_{PV};Entries", 51, 0, 50);
  nDisplacedVertices_hist =
      ibook.book1D("nDisplacedVertices", "Number of Displaced Vertices (Vtx);N_{DV};Entries", 10, 0, 10);
  nDisplacedVerticesNoVtx_hist =
      ibook.book1D("nDisplacedVerticesNoVtx", "Number of Displaced Vertices (NoVtx);N_{DV}^{NoVtx};Entries", 10, 0, 10);
  nMuons_hist = ibook.book1D("nMuons", "Number of Muons (NoVtx);N_{muons};Entries", 10, 0, 10);
  nMuonsVtx_hist = ibook.book1D("nMuonsVtx", "Number of Muons (Vtx);N_{muons}^{Vtx};Entries", 10, 0, 10);
  nElectrons_hist = ibook.book1D("nElectrons", "Number of Electrons;N_{ele};Entries", 10, 0, 10);
  nPhotons_hist = ibook.book1D("nPhotons", "Number of Photons;N_{photon};Entries", 25, 0, 25);
  nPFJets_hist = ibook.book1D("nPFJets", "Number of PF Jets;N_{jet};Entries", 101, 0, 100);
  nPFCands_hist = ibook.book1D("nPFCands", "Number of PF Candidates;N_{pfcand};Entries", 1001, 0, 1000);

  rho_hist = ibook.book1D("rho", "#rho; #rho; Entries", 100, 0.0, 60.0);
  pfMetPhi_hist = ibook.book1D("pfMetPhi", "pf MET #phi; #phi ;Entries", 100, -3.14, 3.14);
  pfMetPt_hist = ibook.book1D("pfMetPt", "pf MET p_{T};p_{T} [GeV];Entries", 100, 0.0, 250.0);

  if (!onlyScouting_) {
    PVvsPU_hist =
        ibook.bookProfile("PVvsPU", "Number of primary vertices vs pile up; pile up; <N_{PV}>", 20, 20, 70, 0, 65);
    rhovsPU_hist = ibook.bookProfile("rhovsPU", "#rho vs pile up; pile up; <#rho>", 20, 20, 70, 0, 45);
  }

  ibook.setCurrentFolder(topfoldername_ + "/PFcand");
  PF_pT_211_hist = ibook.book1DD("pT_posHad", "PF h^{+}  p_{T} (GeV);p_{T} [GeV];Entries", 100, 0.0, 13.0);
  PF_pT_n211_hist = ibook.book1DD("pT_negHad", "PF h^{-} p_{T} (GeV);p_{T} [GeV];Entries", 100, 0.0, 14.0);
  PF_pT_130_hist = ibook.book1DD("pT_neuHad", "PF h^{0} p_{T} (GeV);p_{T} [GeV];Entries", 100, 0.0, 20.0);
  PF_pT_22_hist = ibook.book1DD("pT_gamma", "PF #gamma p_{T} (GeV);p_{T} [GeV];Entries", 100, 0.0, 18.0);
  PF_pT_13_hist = ibook.book1DD("pT_mu_plus", "PF #mu^{+} p_{T} (GeV);p_{T} [GeV];Entries", 100, 0.0, 80.0);
  PF_pT_n13_hist = ibook.book1DD("pT_mu_minus", "PF #mu^{-} p_{T} (GeV);p_{T} [GeV];Entries", 100, 0.0, 80.0);
  PF_pT_2_hist = ibook.book1DD("pT_HF_had", "PF HF h (GeV);p_{T} [GeV];Entries", 100, 0.0, 4.5);
  PF_pT_1_hist = ibook.book1DD("pT_HF_eg", "PF HF e/#gamma p_{T} (GeV);p_{T} [GeV];Entries", 100, 0.0, 6.0);

  PF_eta_211_hist = ibook.book1DD("eta_posHad", "PF h^{+} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_n211_hist = ibook.book1DD("eta_negHad", "PF h^{-} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_130_hist = ibook.book1DD("eta_neuHad", "PF h^{0} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_22_hist = ibook.book1DD("eta_gamma", "PF #gamma #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_13_hist = ibook.book1DD("eta_mu_plus", "PF #mu^{+} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_n13_hist = ibook.book1DD("eta_mu_minus", "PF #mu^{-} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_1_hist = ibook.book1DD("eta_HF_had", "PF HF h #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_2_hist = ibook.book1DD("eta_HF_eg", "PF HF e/#gamma #eta;#eta;Entries", 100, -5.0, 5.0);

  PF_phi_211_hist = ibook.book1DD("phi_posHad", "PF h^{+} #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_n211_hist = ibook.book1DD("phi_negHad", "PF h^{-} #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_130_hist = ibook.book1DD("phi_neuHad", "PF h^{0} #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_22_hist = ibook.book1DD("phi_gamma", "PF #gamma #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_13_hist = ibook.book1DD("phi_mu_plus", "PF #mu^{+} #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_n13_hist = ibook.book1DD("phi_mu_minus", "PF #mu^{-} #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_1_hist = ibook.book1DD("phi_HF_had", "PF HF h #phi;#phi;Entries", 100, -3.2, 3.2);
  PF_phi_2_hist = ibook.book1DD("phi_HF_eg", "PF HF e/#gamma #phi;#phi;Entries", 100, -3.2, 3.2);

  PF_vertex_211_hist =
      ibook.book1DD("vertexIndex_posHad", "PF h^{+} Vertex Index ;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_n211_hist =
      ibook.book1DD("vertexIndex_negHad", "PF h^{-} Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_130_hist =
      ibook.book1DD("vertexIndex_neuHad", "PF h^{0} Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_22_hist = ibook.book1DD("vertexIndex_gamma", "PF #gamma Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_13_hist =
      ibook.book1DD("vertexIndex_mu_plus", "PF #mu^{+} Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_n13_hist =
      ibook.book1DD("vertexIndex_mu_minus", "PF #mu^{-} Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_1_hist = ibook.book1DD("vertexIndex_HF_eg", "PF HF h Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_2_hist =
      ibook.book1DD("vertexIndex_HF_had", "PF HF e/#gamma Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);

  // the following variables make sense only if there is a Track

  PF_normchi2_211_hist =
      ibook.book1DD("normchi2_posHad", "PF h^{+} Norm #chi^{2};Norm #chi^{2};Entries", 100, 0.0, 10.0);
  PF_normchi2_n211_hist =
      ibook.book1DD("normchi2_negHad", "PF h^{-} Norm #chi^{2};Norm #chi^{2};Entries", 100, 0.0, 10.0);
  PF_normchi2_13_hist =
      ibook.book1DD("normchi2_mu_plus", "PF #mu^{+} Norm #chi^{2};Norm #chi^{2};Entries", 100, 0.0, 10.0);
  PF_normchi2_n13_hist =
      ibook.book1DD("normchi2_mu_minus", "PF #mu^{-} Norm #chi^{2};Norm #chi^{2};Entries", 100, 0.0, 10.0);

  PF_dz_211_hist = ibook.book1DD("dz_posHad", "PF h^{+} d_{z} (cm);d_{z} (cm);Entries", 100, -1.0, 1.0);
  PF_dz_n211_hist = ibook.book1DD("dz_negHad", "PF h^{-} d_{z} (cm);d_{z} (cm);Entries", 100, -1.0, 1.0);
  PF_dz_13_hist = ibook.book1DD("dz_mu_plus", "PF #mu^{+} d_{z} (cm);d_{z} (cm);Entries", 100, -1.0, 1.0);
  PF_dz_n13_hist = ibook.book1DD("dz_mu_minus", "PF #mu^{-} d_{z} (cm);d_{z} (cm);Entries", 100, -1.0, 1.0);

  PF_dxy_211_hist = ibook.book1DD("dxy_posHad", "PF h^{+} d_{xy} (cm);d_{xy} (cm);Entries", 100, -0.5, 0.5);
  PF_dxy_n211_hist = ibook.book1DD("dxy_negHad", "PF h^{-} d_{xy} (cm);d_{xy} (cm);Entries", 100, -0.5, 0.5);
  PF_dxy_13_hist = ibook.book1DD("dxy_mu_plus", "PF #mu^{+} d_{xy} (cm);d_{xy} (cm);Entries", 100, -0.5, 0.5);
  PF_dxy_n13_hist = ibook.book1DD("dxy_mu_minus", "PF #mu^{-} d_{xy} (cm);d_{xy} (cm);Entries", 100, -0.5, 0.5);

  PF_dzsig_211_hist =
      ibook.book1DD("dzsig_posHad", "PF h^{+} d_{z} Signficance;d_{z}/#sigma_{dz};Entries", 100, 0.0, 10.0);
  PF_dzsig_n211_hist =
      ibook.book1DD("dzsig_negHad", "PF h^{-} d_{z} Signficance;d_{z}/#sigma_{dz};Entries", 100, 0.0, 10.0);
  PF_dzsig_13_hist =
      ibook.book1DD("dzsig_mu_plus", "PF #mu^{+} d_{z} Signficance;d_{z}/#sigma_{dz};Entries", 100, 0.0, 10.0);
  PF_dzsig_n13_hist =
      ibook.book1DD("dzsig_mu_minus", "PF #mu^{-} d_{z} Signficance;d_{z}/#sigma_{dz};Entries", 100, 0.0, 10.0);

  PF_dxysig_211_hist =
      ibook.book1DD("dxysig_posHad", "PF h^{+} d_{xy} Significance;d_{xy}/#sigma_{dxy};Entries", 100, 0.0, 10.0);
  PF_dxysig_n211_hist =
      ibook.book1DD("dxysig_negHad", "PF h^{-} d_{xy} Significance;d_{xy}/#sigma_{dxy};Entries", 100, 0.0, 10.0);
  PF_dxysig_13_hist =
      ibook.book1DD("dxysig_mu_plus", "PF #mu^{+} d_{xy} Significance;d_{xy}/#sigma_{dxy};Entries", 100, 0.0, 10.0);
  PF_dxysig_n13_hist =
      ibook.book1DD("dxysig_mu_minus", "PF #mu^{-} d_{xy} Significance;d_{xy}/#sigma_{dxy};Entries", 100, 0.0, 10.0);

  // These variables are actually the difference between the PF candidate reconstructed kinematics and it's bestTrack ones.
  // This behaviour is governed by the "relativeTrackVars" parameter of HLTScoutingPFProducer
  // see https://github.com/cms-sw/cmssw/blob/master/HLTrigger/JetMET/plugins/HLTScoutingPFProducer.cc#L177-L185no

  PF_trk_pt_211_hist =
      ibook.book1DD("trk_pt_posHad",
                    "PF h^{+} #Delta p_{T}(Track - Cand) (GeV);#Delta p_{T}(Track - Cand) (GeV);Entries",
                    100,
                    -0.01,
                    0.01);
  PF_trk_pt_n211_hist =
      ibook.book1DD("trk_pt_negHad",
                    "PF h^{-} #Delta p_{T}(Track - Cand) (GeV);#Delta p_{T}(Track - Cand) (GeV);Entries",
                    100,
                    -0.01,
                    0.01);
  PF_trk_pt_13_hist =
      ibook.book1DD("trk_pt_mu_plus",
                    "PF #mu^{+} #Delta p_{T}(Track - Cand) (GeV);#Delta p_{T}(Track - Cand) (GeV);Entries",
                    100,
                    -0.01,
                    0.01);
  PF_trk_pt_n13_hist =
      ibook.book1DD("trk_pt_mu_minus",
                    "PF #mu^{-} #Delta p_{T}(Track - Cand) (GeV);#Delta p_{T}(Track - Cand) (GeV);Entries",
                    100,
                    -0.01,
                    0.01);

  PF_trk_eta_211_hist = ibook.book1DD(
      "trk_eta_posHad", "PF h^{+} #Delta #eta(Track - Cand);#Delta #eta(Track - Cand);Entries", 100, -0.01, 0.01);
  PF_trk_eta_n211_hist = ibook.book1DD(
      "trk_eta_negHad", "PF h^{-} #Delta #eta(Track - Cand);#Delta #eta(Track - Cand);Entries", 100, -0.01, 0.01);
  PF_trk_eta_13_hist = ibook.book1DD(
      "trk_eta_mu_plus", "PF #mu^{+} #Delta #eta(Track - Cand);#Delta #eta(Track - Cand);Entries", 100, -0.01, 0.01);
  PF_trk_eta_n13_hist = ibook.book1DD(
      "trk_eta_mu_minus", "PF #mu^{-} #Delta #eta(Track - Cand);#Delta #eta(Track - Cand);Entries", 100, -0.01, 0.01);

  PF_trk_phi_211_hist = ibook.book1DD(
      "trk_phi_posHad", "PF h^{+} #Delta #phi(Track - Cand);#Delta #phi(Track - Cand) [rad];Entries", 100, -0.01, 0.01);
  PF_trk_phi_n211_hist = ibook.book1DD(
      "trk_phi_negHad", "PF h^{-} #Delta #phi(Track - Cand);#Delta #phi(Track - Cand) [rad];Entries", 100, -0.01, 0.01);
  PF_trk_phi_13_hist = ibook.book1DD("trk_phi_mu_plus",
                                     "PF #mu^{+} #Delta #phi(Track - Cand);#Delta #phi(Track - Cand) [rad];Entries",
                                     100,
                                     -0.01,
                                     0.01);
  PF_trk_phi_n13_hist = ibook.book1DD("trk_phi_mu_minus",
                                      "PF #mu^{-} #Delta #phi(Track - Cand);#Delta #phi(Track - Cand) [rad];Entries",
                                      100,
                                      -0.01,
                                      0.01);

  ibook.setCurrentFolder(topfoldername_ + "/Photon");
  pt_pho_hist = ibook.book1DD("pt_pho", "Photon p_{T}; p_{T} (GeV); Entries", 100, 0.0, 100.0);
  eta_pho_hist = ibook.book1DD("eta_pho", "photon #eta; #eta; Entries", 100, -2.7, 2.7);
  phi_pho_hist = ibook.book1DD("phi_pho", "Photon #phi; #phi (rad); Entries", 100, -3.14, 3.14);
  rawEnergy_pho_hist = ibook.book1DD("rawEnergy_pho", "Raw Energy Photon; Energy (GeV); Entries", 100, 0.0, 250.0);
  preshowerEnergy_pho_hist =
      ibook.book1DD("preshowerEnergy_pho", "Preshower Energy Photon; Energy (GeV); Entries", 100, 0.0, 8.0);
  corrEcalEnergyError_pho_hist = ibook.book1DD(
      "corrEcalEnergyError_pho", "Corrected ECAL Energy Error Photon; Energy Error (GeV); Entries", 100, 0.0, 20.0);
  sigmaIetaIeta_pho_hist =
      ibook.book1DD("sigmaIetaIeta_pho", "Sigma iEta iEta Photon; #sigma_{i#eta i#eta}; Entries", 100, 0.0, 0.5);
  hOverE_pho_hist = ibook.book1DD("hOverE_pho", "H/E Photon; H/E; Entries", 100, 0.0, 1.5);
  ecalIso_pho_hist = ibook.book1DD("ecalIso_pho", "ECAL Isolation Photon; Isolation (GeV); Entries", 100, 0.0, 100.0);
  hcalIso_pho_hist = ibook.book1DD("hcalIso_pho", "HCAL Isolation Photon; Isolation (GeV); Entries", 100, 0.0, 100.0);
  trackIso_pho_hist = ibook.book1DD("trackIso_pho", "Track Isolation Photon; Isolation (GeV); Entries", 100, 0.0, 0.05);
  r9_pho_hist = ibook.book1DD("r9_pho", "R9; R9; Entries", 100, 0.0, 5);
  sMin_pho_hist = ibook.book1DD("sMin_pho", "sMin Photon; sMin; Entries", 100, 0.0, 3);
  sMaj_pho_hist = ibook.book1DD("sMaj_pho", "sMaj Photon; sMaj; Entries", 100, 0.0, 3);

  ibook.setCurrentFolder(topfoldername_ + "/Electron");
  pt_ele_hist = ibook.book1DD("pt_ele", "Electron p_{T}; p_{T} (GeV); Entries", 100, 0.0, 100.0);
  eta_ele_hist = ibook.book1DD("eta_ele", "Electron #eta; #eta; Entries", 100, -2.7, 2.7);
  phi_ele_hist = ibook.book1DD("phi_ele", "Electron #phi; #phi (rad); Entries", 100, -3.14, 3.14);
  rawEnergy_ele_hist = ibook.book1DD("rawEnergy_ele", "Raw Energy Electron; Energy (GeV); Entries", 100, 0.0, 250.0);
  preshowerEnergy_ele_hist =
      ibook.book1DD("preshowerEnergy_ele", "Preshower Energy Electron; Energy (GeV); Entries", 100, 0.0, 10.0);
  corrEcalEnergyError_ele_hist = ibook.book1DD(
      "corrEcalEnergyError_ele", "Corrected ECAL Energy Error Electron; Energy Error (GeV); Entries", 100, 0.0, 20.0);
  dEtaIn_ele_hist = ibook.book1DD("dEtaIn_ele", "dEtaIn Electron; dEtaIn; Entries", 100, -0.05, 0.05);
  dPhiIn_ele_hist = ibook.book1DD("dPhiIn_ele", "dPhiIn Electron; dPhiIn; Entries", 100, -0.5, 0.5);
  sigmaIetaIeta_ele_hist =
      ibook.book1DD("sigmaIetaIeta_ele", "Sigma iEta iEta Electron; #sigma_{i#eta i#eta}; Entries", 100, 0.0, 0.05);
  hOverE_ele_hist = ibook.book1DD("hOverE_ele", "H/E Electron; H/E; Entries", 100, 0.0, 0.3);
  ooEMOop_ele_hist = ibook.book1DD("ooEMOop_ele", "1/E - 1/p Electron; 1/E - 1/p; Entries", 100, -0.3, 0.3);
  missingHits_ele_hist = ibook.book1DD("missingHits_ele", "Missing Hits Electron; Count; Entries", 10, 0, 5);
  trackfbrem_ele_hist = ibook.book1DD("trackfbrem_ele", "Track fbrem Electron; fbrem; Entries", 100, -1.5, 1.0);
  ecalIso_ele_hist = ibook.book1DD("ecalIso_ele", "ECAL Isolation Electron; Isolation (GeV); Entries", 100, 0.0, 70.0);
  hcalIso_ele_hist = ibook.book1DD("hcalIso_ele", "HCAL Isolation Electron; Isolation (GeV); Entries", 100, 0.0, 60.0);
  trackIso_ele_hist =
      ibook.book1DD("trackIso_ele", "Track Isolation Electron; Isolation (GeV); Entries", 100, 0.0, 0.05);
  r9_ele_hist = ibook.book1DD("r9_ele", "R9 Electron; R9; Entries", 100, 0.0, 5);
  sMin_ele_hist = ibook.book1DD("sMin_ele", "sMin Electron; sMin; Entries", 100, 0.0, 3);
  sMaj_ele_hist = ibook.book1DD("sMaj_ele", "sMaj Electron; sMaj; Entries", 100, 0.0, 3);

  // book the muon histograms (noVtx and Vtx collections)
  const std::array<std::string, 2> muonLabels = {{"muonsNoVtx", "muonsVtx"}};
  const std::array<std::string, 2> suffixes = {{"_noVtx", "_Vtx"}};
  for (int i = 0; i < 2; ++i) {
    ibook.setCurrentFolder(topfoldername_ + "/" + muonLabels[i]);

    const std::string& sfx = suffixes[i];
    const std::string& lbl = muonLabels[i];

    pt_mu_hist[i] = ibook.book1DD("pt_mu" + sfx, "Muon p_{T} (" + lbl + "); p_{T} (GeV); Entries", 100, 0.0, 200.0);
    eta_mu_hist[i] = ibook.book1DD("eta_mu" + sfx, "Muon #eta (" + lbl + "); #eta; Entries", 100, -2.7, 2.7);
    phi_mu_hist[i] = ibook.book1DD("phi_mu" + sfx, "Muon #phi (" + lbl + "); #phi (rad); Entries", 100, -3.14, 3.14);
    type_mu_hist[i] = ibook.book1DD("type_mu" + sfx, "Muon Type (" + lbl + "); Type; Entries", 10, 0, 10);
    charge_mu_hist[i] = ibook.book1DD("charge_mu" + sfx, "Muon Charge (" + lbl + "); Charge; Entries", 3, -1, 2);
    normalizedChi2_mu_hist[i] =
        ibook.book1DD("normalizedChi2_mu" + sfx, "Normalized chi2 (" + lbl + "); chi2; Entries", 100, 0.0, 10.0);
    ecalIso_mu_hist[i] = ibook.book1DD(
        "ecalIso_mu" + sfx, "ECAL Isolation Muon (" + lbl + "); Isolation (GeV); Entries", 100, 0.0, 100.0);
    hcalIso_mu_hist[i] = ibook.book1DD(
        "hcalIso_mu" + sfx, "HCAL Isolation Muon (" + lbl + "); Isolation (GeV); Entries", 100, 0.0, 100.0);
    trackIso_mu_hist[i] = ibook.book1DD(
        "trackIso_mu" + sfx, "Track Isolation Muon (" + lbl + "); Isolation (GeV); Entries", 100, 0.0, 10.0);
    nValidStandAloneMuonHits_mu_hist[i] = ibook.book1DD(
        "nValidStandAloneMuonHits_mu" + sfx, "Valid Standalone Muon Hits (" + lbl + "); Hits; Entries", 50, 0, 50);
    nStandAloneMuonMatchedStations_mu_hist[i] =
        ibook.book1DD("nStandAloneMuonMatchedStations_mu" + sfx,
                      "Standalone Muon Matched Stations (" + lbl + "); Stations; Entries",
                      10,
                      0,
                      10);
    nValidRecoMuonHits_mu_hist[i] =
        ibook.book1DD("nValidRecoMuonHits_mu" + sfx, "Valid Reco Muon Hits (" + lbl + "); Hits; Entries", 50, 0, 50);
    nRecoMuonChambers_mu_hist[i] =
        ibook.book1DD("nRecoMuonChambers_mu" + sfx, "Reco Muon Chambers (" + lbl + "); Chambers; Entries", 10, 0, 20);
    nRecoMuonChambersCSCorDT_mu_hist[i] =
        ibook.book1DD("nRecoMuonChambersCSCorDT_mu" + sfx,
                      "Reco Muon Chambers (CSC or DT) (" + lbl + "); Chambers; Entries",
                      10,
                      0,
                      14);
    nRecoMuonMatches_mu_hist[i] =
        ibook.book1DD("nRecoMuonMatches_mu" + sfx, "Reco Muon Matches (" + lbl + "); Matches; Entries", 10, 0, 10);
    nRecoMuonMatchedStations_mu_hist[i] = ibook.book1DD(
        "nRecoMuonMatchedStations_mu" + sfx, "Reco Muon Matched Stations (" + lbl + "); Stations; Entries", 10, 0, 10);
    nRecoMuonExpectedMatchedStations_mu_hist[i] =
        ibook.book1DD("nRecoMuonExpectedMatchedStations_mu" + sfx,
                      "Reco Muon Expected Matched Stations (" + lbl + "); Stations; Entries",
                      10,
                      0,
                      10);
    recoMuonStationMask_mu_hist[i] =
        ibook.book1DD("recoMuonStationMask_mu" + sfx, "Reco Muon Station Mask (" + lbl + "); Mask; Entries", 20, 0, 20);
    nRecoMuonMatchedRPCLayers_mu_hist[i] = ibook.book1DD(
        "nRecoMuonMatchedRPCLayers_mu" + sfx, "Reco Muon Matched RPC Layers (" + lbl + "); Layers; Entries", 10, 0, 2);
    recoMuonRPClayerMask_mu_hist[i] = ibook.book1DD(
        "recoMuonRPClayerMask_mu" + sfx, "Reco Muon RPC Layer Mask (" + lbl + "); Mask; Entries", 20, 0, 5);
    nValidPixelHits_mu_hist[i] =
        ibook.book1DD("nValidPixelHits_mu" + sfx, "Valid Pixel Hits (" + lbl + "); Hits; Entries", 20, 0, 20);
    nValidStripHits_mu_hist[i] =
        ibook.book1DD("nValidStripHits_mu" + sfx, "Valid Strip Hits (" + lbl + "); Hits; Entries", 50, 0, 50);
    nPixelLayersWithMeasurement_mu_hist[i] =
        ibook.book1DD("nPixelLayersWithMeasurement_mu" + sfx,
                      "Pixel Layers with Measurement (" + lbl + "); Layers; Entries",
                      10,
                      0,
                      10);
    nTrackerLayersWithMeasurement_mu_hist[i] =
        ibook.book1DD("nTrackerLayersWithMeasurement_mu" + sfx,
                      "Tracker Layers with Measurement (" + lbl + "); Layers; Entries",
                      20,
                      0,
                      20);
    trk_chi2_mu_hist[i] =
        ibook.book1DD("trk_chi2_mu" + sfx, "Muon Tracker chi2 (" + lbl + "); #chi^{2}; Entries", 100, 0.0, 100.0);
    trk_ndof_mu_hist[i] =
        ibook.book1DD("trk_ndof_mu" + sfx, "Muon Tracker Ndof (" + lbl + "); Ndof; Entries", 100, 0, 100);
    trk_dxy_mu_hist[i] =
        ibook.book1DD("trk_dxy_mu" + sfx, "Muon Tracker d_{xy} (" + lbl + "); d_{xy} (cm); Entries", 100, -0.5, 0.5);
    trk_dz_mu_hist[i] =
        ibook.book1DD("trk_dz_mu" + sfx, "Muon Tracker d_{z} (" + lbl + "); d_{z} (cm); Entries", 100, -20.0, 20.0);
    trk_qoverp_mu_hist[i] = ibook.book1DD("trk_qoverp_mu" + sfx, "Muon q/p (" + lbl + "); q/p; Entries", 100, -1, 1);
    trk_lambda_mu_hist[i] =
        ibook.book1DD("trk_lambda_mu" + sfx, "Muon Lambda (" + lbl + "); #lambda; Entries", 100, -2, 2);
    trk_pt_mu_hist[i] =
        ibook.book1DD("trk_pt_mu" + sfx, "Muon Tracker p_{T} (" + lbl + "); p_{T} (GeV); Entries", 100, 0.0, 200.0);
    trk_phi_mu_hist[i] =
        ibook.book1DD("trk_phi_mu" + sfx, "Muon Tracker #phi (" + lbl + "); #phi (rad); Entries", 100, -3.14, 3.14);
    trk_eta_mu_hist[i] =
        ibook.book1DD("trk_eta_mu" + sfx, "Muon Tracker #eta (" + lbl + "); #eta; Entries", 100, -2.7, 2.7);
    trk_dxyError_mu_hist[i] = ibook.book1DD(
        "trk_dxyError_mu" + sfx, "Muon d_{xy} Error (" + lbl + "); d_{xy} Error (cm); Entries", 100, 0.0, 0.05);
    trk_dzError_mu_hist[i] = ibook.book1DD(
        "trk_dzError_mu" + sfx, "Muon d_{z} Error (" + lbl + "); d_{z} Error (cm); Entries", 100, 0.0, 0.05);
    trk_qoverpError_mu_hist[i] =
        ibook.book1DD("trk_qoverpError_mu" + sfx, "Muon q/p Error (" + lbl + "); q/p Error; Entries", 100, 0.0, 0.01);
    trk_lambdaError_mu_hist[i] = ibook.book1DD(
        "trk_lambdaError_mu" + sfx, "Muon Lambda Error (" + lbl + "); #lambda Error; Entries", 100, 0.0, 0.1);
    trk_phiError_mu_hist[i] = ibook.book1DD(
        "trk_phiError_mu" + sfx, "Muon Phi Error (" + lbl + "); #phi Error (rad); Entries", 100, 0.0, 0.01);
    trk_dsz_mu_hist[i] = ibook.book1DD("trk_dsz_mu" + sfx, "Muon dsz (" + lbl + "); dsz (cm); Entries", 100, -2, 2);
    trk_dszError_mu_hist[i] =
        ibook.book1DD("trk_dszError_mu" + sfx, "Muon dsz Error (" + lbl + "); dsz Error (cm); Entries", 100, 0.0, 0.05);
    trk_qoverp_lambda_cov_mu_hist[i] = ibook.book1DD("trk_qoverp_lambda_cov_mu" + sfx,
                                                     "Muon q/p-#lambda Covariance (" + lbl + "); Covariance; Entries",
                                                     100,
                                                     -0.001,
                                                     0.001);
    trk_qoverp_phi_cov_mu_hist[i] = ibook.book1DD("trk_qoverp_phi_cov_mu" + sfx,
                                                  "Muon q/p-#phi Covariance (" + lbl + "); Covariance; Entries",
                                                  100,
                                                  -0.001,
                                                  0.001);
    trk_qoverp_dxy_cov_mu_hist[i] = ibook.book1DD("trk_qoverp_dxy_cov_mu" + sfx,
                                                  "Muon q/p-d_{xy} Covariance (" + lbl + "); Covariance; Entries",
                                                  100,
                                                  -0.001,
                                                  0.001);
    trk_qoverp_dsz_cov_mu_hist[i] = ibook.book1DD(
        "trk_qoverp_dsz_cov_mu" + sfx, "Muon q/p-dsz Covariance (" + lbl + "); Covariance; Entries", 100, -0.001, 0.001);
    trk_lambda_phi_cov_mu_hist[i] = ibook.book1DD("trk_lambda_phi_cov_mu" + sfx,
                                                  "Muon Lambda-#phi Covariance (" + lbl + "); Covariance; Entries",
                                                  100,
                                                  -0.001,
                                                  0.001);
    trk_lambda_dxy_cov_mu_hist[i] = ibook.book1DD("trk_lambda_dxy_cov_mu" + sfx,
                                                  "Muon Lambda-d_{xy} Covariance (" + lbl + "); Covariance; Entries",
                                                  100,
                                                  -0.001,
                                                  0.001);
    trk_lambda_dsz_cov_mu_hist[i] = ibook.book1DD("trk_lambda_dsz_cov_mu" + sfx,
                                                  "Muon Lambda-dsz Covariance (" + lbl + "); Covariance; Entries",
                                                  100,
                                                  -0.001,
                                                  0.001);
    trk_phi_dxy_cov_mu_hist[i] = ibook.book1DD(
        "trk_phi_dxy_cov_mu" + sfx, "Muon Phi-d_{xy} Covariance (" + lbl + "); Covariance; Entries", 100, -0.001, 0.001);
    trk_phi_dsz_cov_mu_hist[i] = ibook.book1DD(
        "trk_phi_dsz_cov_mu" + sfx, "Muon Phi-dsz Covariance (" + lbl + "); Covariance; Entries", 100, -0.001, 0.001);
    trk_dxy_dsz_cov_mu_hist[i] = ibook.book1DD(
        "trk_dxy_dsz_cov_mu" + sfx, "Muon d_{xy}-dsz Covariance (" + lbl + "); Covariance; Entries", 100, -0.001, 0.001);
    trk_vx_mu_hist[i] =
        ibook.book1DD("trk_vx_mu" + sfx, "Muon Tracker Vertex X (" + lbl + "); x (cm); Entries", 100, -0.5, 0.5);
    trk_vy_mu_hist[i] =
        ibook.book1DD("trk_vy_mu" + sfx, "Muon Tracker Vertex Y (" + lbl + "); y (cm); Entries", 100, -0.5, 0.5);
    trk_vz_mu_hist[i] =
        ibook.book1DD("trk_vz_mu" + sfx, "Muon Tracker Vertex Z (" + lbl + "); z (cm); Entries", 100, -20.0, 20.0);
  }

  ibook.setCurrentFolder(topfoldername_ + "/PFJet");
  pt_pfj_hist = ibook.book1DD("pt_pfj", "PF Jet p_{T}; p_{T} (GeV); Entries", 100, 0.0, 150.0);
  eta_pfj_hist = ibook.book1DD("eta_pfj", "PF Jet #eta; #eta; Entries", 100, -5.0, 5.0);
  phi_pfj_hist = ibook.book1DD("phi_pfj", "PF Jet #phi; #phi (rad); Entries", 100, -3.14, 3.14);
  m_pfj_hist = ibook.book1DD("m_pfj", "PF Jet Mass; Mass (GeV); Entries", 100, 0.0, 40.0);
  jetArea_pfj_hist = ibook.book1DD("jetArea_pfj", "PF Jet Area; Area; Entries", 100, 0.0, 0.8);
  chargedHadronEnergy_pfj_hist =
      ibook.book1DD("chargedHadronEnergy_pfj", "Charged Hadron Energy; Energy (GeV); Entries", 100, 0.0, 150.0);
  neutralHadronEnergy_pfj_hist =
      ibook.book1DD("neutralHadronEnergy_pfj", "Neutral Hadron Energy; Energy (GeV); Entries", 100, 0.0, 600.0);
  photonEnergy_pfj_hist = ibook.book1DD("photonEnergy_pfj", "Photon Energy; Energy (GeV); Entries", 100, 0.0, 90.0);
  electronEnergy_pfj_hist =
      ibook.book1DD("electronEnergy_pfj", "Electron Energy; Energy (GeV); Entries", 100, 0.0, 3.0);
  muonEnergy_pfj_hist = ibook.book1DD("muonEnergy_pfj", "Muon Energy; Energy (GeV); Entries", 100, 0.0, 3.0);
  HFHadronEnergy_pfj_hist =
      ibook.book1DD("HFHadronEnergy_pfj", "HF Hadron Energy; Energy (GeV); Entries", 100, 0.0, 300.0);
  HFEMEnergy_pfj_hist = ibook.book1DD("HFEMEnergy_pfj", "HF EM Energy; Energy (GeV); Entries", 100, 0.0, 300.0);
  chargedHadronMultiplicity_pfj_hist =
      ibook.book1DD("chargedHadronMultiplicity_pfj", "Charged Hadron Multiplicity; Multiplicity; Entries", 50, 0, 25);
  neutralHadronMultiplicity_pfj_hist =
      ibook.book1DD("neutralHadronMultiplicity_pfj", "Neutral Hadron Multiplicity; Multiplicity; Entries", 50, 0, 10);
  photonMultiplicity_pfj_hist =
      ibook.book1DD("photonMultiplicity_pfj", "Photon Multiplicity; Multiplicity; Entries", 50, 0, 22);
  electronMultiplicity_pfj_hist =
      ibook.book1DD("electronMultiplicity_pfj", "Electron Multiplicity; Multiplicity; Entries", 20, 0, 5);
  muonMultiplicity_pfj_hist =
      ibook.book1DD("muonMultiplicity_pfj", "Muon Multiplicity; Multiplicity; Entries", 20, 0, 5);
  HFHadronMultiplicity_pfj_hist =
      ibook.book1DD("HFHadronMultiplicity_pfj", "HF Hadron Multiplicity; Multiplicity; Entries", 20, 0, 20);
  HFEMMultiplicity_pfj_hist =
      ibook.book1DD("HFEMMultiplicity_pfj", "HF EM Multiplicity; Multiplicity; Entries", 20, 0, 20);
  HOEnergy_pfj_hist = ibook.book1DD("HOEnergy_pfj", "HO Energy; Energy (GeV); Entries", 100, 0.0, 5.0);
  mvaDiscriminator_pfj_hist =
      ibook.book1DD("mvaDiscriminator_pfj", "MVA Discriminator; Score; Entries", 100, -1.0, 1.0);

  ibook.setCurrentFolder(topfoldername_ + "/PrimaryVertex");
  x_pv_hist = ibook.book1DD("x_pv", "Primary Vertex X Position; x (cm); Entries", 100, -0.5, 0.5);
  y_pv_hist = ibook.book1DD("y_pv", "Primary Vertex Y Position; y (cm); Entries", 100, -0.5, 0.5);
  z_pv_hist = ibook.book1DD("z_pv", "Primary Vertex Z Position; z (cm); Entries", 100, -20.0, 20.0);
  zError_pv_hist = ibook.book1DD("zError_pv", "Primary Vertex Z Error; z Error (cm); Entries", 100, 0.0, 0.05);
  xError_pv_hist = ibook.book1DD("xError_pv", "Primary Vertex X Error; x Error (cm); Entries", 100, 0.0, 0.05);
  yError_pv_hist = ibook.book1DD("yError_pv", "Primary Vertex Y Error; y Error (cm); Entries", 100, 0.0, 0.05);
  tracksSize_pv_hist =
      ibook.book1DD("tracksSize_pv", "Number of Tracks at Primary Vertex; Tracks; Entries", 100, 0, 100);
  chi2_pv_hist = ibook.book1DD("chi2_pv", "Primary Vertex chi2; #chi^{2}; Entries", 100, 0.0, 50.0);
  ndof_pv_hist = ibook.book1DD("ndof_pv", "Primary Vertex Ndof; Ndof; Entries", 100, 0, 100);
  isValidVtx_pv_hist =
      ibook.book1DD("isValidVtx_pv", "Is Valid Primary Vertex?; 0 = False, 1 = True; Entries", 2, 0, 2);
  xyCov_pv_hist = ibook.book1DD("xyCov_pv", "Primary Vertex XY Covariance; Cov(x,y); Entries", 100, -0.01, 0.01);
  xzCov_pv_hist = ibook.book1DD("xzCov_pv", "Primary Vertex XZ Covariance; Cov(x,z); Entries", 100, -0.01, 0.01);
  yzCov_pv_hist = ibook.book1DD("yzCov_pv", "Primary Vertex YZ Covariance; Cov(y,z); Entries", 100, -0.01, 0.01);

  // book the displaced vertex histograms (Vtx and noVtx collections)
  const std::array<std::string, 2> vertexLabels = {{"displacedVertices", "displacedVerticesNoVtx"}};
  const std::array<std::string, 2> suffixesVtx = {{"_Vtx", "_noVtx"}};

  for (int i = 0; i < 2; ++i) {
    const std::string& sfx = suffixesVtx[i];
    const std::string& lbl = vertexLabels[i];

    ibook.setCurrentFolder(topfoldername_ + "/" + vertexLabels[i]);

    x_vtx_hist[i] = ibook.book1DD("x_vtx" + sfx, "Vertex X Position (" + lbl + "); x (cm); Entries", 100, -0.5, 0.5);
    y_vtx_hist[i] = ibook.book1DD("y_vtx" + sfx, "Vertex Y Position (" + lbl + "); y (cm); Entries", 100, -0.5, 0.5);
    z_vtx_hist[i] = ibook.book1DD("z_vtx" + sfx, "Vertex Z Position (" + lbl + "); z (cm); Entries", 100, -20.0, 20.0);
    xError_vtx_hist[i] =
        ibook.book1DD("xError_vtx" + sfx, "Vertex X Error (" + lbl + "); x Error (cm); Entries", 100, 0.0, 0.2);
    yError_vtx_hist[i] =
        ibook.book1DD("yError_vtx" + sfx, "Vertex Y Error (" + lbl + "); y Error (cm); Entries", 100, 0.0, 0.2);
    zError_vtx_hist[i] =
        ibook.book1DD("zError_vtx" + sfx, "Vertex Z Error (" + lbl + "); z Error (cm); Entries", 100, 0.0, 0.2);
    tracksSize_vtx_hist[i] =
        ibook.book1DD("tracksSize_vtx" + sfx, "Number of Tracks at Vertex (" + lbl + "); Tracks; Entries", 100, 0, 100);
    chi2_vtx_hist[i] =
        ibook.book1DD("chi2_vtx" + sfx, "Vertex #chi^{2} (" + lbl + "); #chi^{2}; Entries", 100, 0.0, 5.0);
    ndof_vtx_hist[i] = ibook.book1DD("ndof_vtx" + sfx, "Vertex Ndof (" + lbl + "); Ndof; Entries", 100, 0, 5);
    isValidVtx_vtx_hist[i] =
        ibook.book1DD("isValidVtx_vtx" + sfx, "Is Valid Vertex? (" + lbl + "); 0 = False, 1 = True; Entries", 2, 0, 2);
    xyCov_vtx_hist[i] =
        ibook.book1DD("xyCov_vtx" + sfx, "Vertex XY Covariance (" + lbl + "); Cov(x,y); Entries", 100, -0.01, 0.01);
    xzCov_vtx_hist[i] =
        ibook.book1DD("xzCov_vtx" + sfx, "Vertex XZ Covariance (" + lbl + "); Cov(x,z); Entries", 100, -0.01, 0.01);
    yzCov_vtx_hist[i] =
        ibook.book1DD("yzCov_vtx" + sfx, "Vertex YZ Covariance (" + lbl + "); Cov(y,z); Entries", 100, -0.01, 0.01);
  }

  ibook.setCurrentFolder(topfoldername_ + "/Tracking");
  tk_pt_tk_hist = ibook.book1DD("tk_pt_tk", "Track p_{T}; p_{T} (GeV); Entries", 100, 0.0, 30.0);
  tk_eta_tk_hist = ibook.book1DD("tk_eta_tk", "Track #eta; #eta; Entries", 100, -2.7, 2.7);
  tk_phi_tk_hist = ibook.book1DD("tk_phi_tk", "Track #phi; #phi (rad); Entries", 100, -3.14, 3.14);
  tk_chi2_tk_hist = ibook.book1DD("tk_chi2_tk", "Track #chi^{2}; #chi^{2}; Entries", 100, 0.0, 50.0);
  tk_ndof_tk_hist = ibook.book1DD("tk_ndof_tk", "Track Ndof; Ndof; Entries", 100, 0, 10);
  tk_charge_tk_hist = ibook.book1DD("tk_charge_tk", "Track Charge; Charge; Entries", 3, -1, 2);
  tk_dxy_tk_hist = ibook.book1DD("tk_dxy_tk", "Track d_{xy}; d_{xy} (cm); Entries", 100, -0.5, 0.5);
  tk_dz_tk_hist = ibook.book1DD("tk_dz_tk", "Track d_{z}; d_{z} (cm); Entries", 100, -20.0, 20.0);
  tk_nValidPixelHits_tk_hist = ibook.book1DD("tk_nValidPixelHits_tk", "Valid Pixel Hits; Hits; Entries", 20, 0, 20);
  tk_nTrackerLayersWithMeasurement_tk_hist = ibook.book1DD(
      "tk_nTrackerLayersWithMeasurement_tk", "Tracker Layers with Measurement; Layers; Entries", 20, 0, 20);
  tk_nValidStripHits_tk_hist = ibook.book1DD("tk_nValidStripHits_tk", "Valid Strip Hits; Hits; Entries", 50, 0, 50);
  tk_qoverp_tk_hist = ibook.book1DD("tk_qoverp_tk", "q/p; q/p; Entries", 100, -1.0, 1.0);
  tk_lambda_tk_hist = ibook.book1DD("tk_lambda_tk", "Lambda; #lambda; Entries", 100, -2, 2);
  tk_dxy_Error_tk_hist = ibook.book1DD("tk_dxy_Error_tk", "d_{xy} Error; d_{xy} Error (cm); Entries", 100, 0.0, 0.05);
  tk_dz_Error_tk_hist = ibook.book1DD("tk_dz_Error_tk", "d_{z} Error; d_{z} Error (cm); Entries", 100, 0.0, 0.05);
  tk_qoverp_Error_tk_hist = ibook.book1DD("tk_qoverp_Error_tk", "q/p Error; q/p Error; Entries", 100, 0.0, 0.05);
  tk_lambda_Error_tk_hist = ibook.book1DD("tk_lambda_Error_tk", "Lambda Error; #lambda Error; Entries", 100, 0.0, 0.1);
  tk_phi_Error_tk_hist = ibook.book1DD("tk_phi_Error_tk", "Phi Error; #phi Error (rad); Entries", 100, 0.0, 0.01);
  tk_dsz_tk_hist = ibook.book1DD("tk_dsz_tk", "dsz; dsz (cm); Entries", 100, -2, 2);
  tk_dsz_Error_tk_hist = ibook.book1DD("tk_dsz_Error_tk", "dsz Error; dsz Error (cm); Entries", 100, 0.0, 0.05);
  tk_vtxInd_tk_hist = ibook.book1DD("tk_vtxInd_tk", "Vertex Index; Index; Entries", 50, 0, 50);
  tk_vx_tk_hist = ibook.book1DD("tk_vx_tk", "Tracker Vertex X; x (cm); Entries", 100, -0.5, 0.5);
  tk_vy_tk_hist = ibook.book1DD("tk_vy_tk", "Tracker Vertex Y; y (cm); Entries", 100, -0.5, 0.5);
  tk_vz_tk_hist = ibook.book1DD("tk_vz_tk", "Tracker Vertex Z; z (cm); Entries", 100, -20.0, 20.0);
  tk_chi2_ndof_tk_hist = ibook.book1DD("tk_chi2_ndof_tk", "Reduced #chi^{2}; #chi^{2}/NDOF; Entries", 100, 0, 50);
  tk_chi2_prob_hist = ibook.book1DD("tk_chi2_prob_hist", "p(#chi^{2}, NDOF); p(#chi^{2}, NDOF); Entries", 100, 0, 1);
  tk_PV_dz_hist = ibook.book1DD("tk_PV_dz", "Track d_{z} w.r.t. PV; Track d_{z} w.r.t. PV; Entries", 100, -0.35, 0.35);
  tk_PV_dxy_hist =
      ibook.book1DD("tk_PV_dxy", "Track d_{xy} w.r.t. PV; Track d_{xy} w.r.t. PV; Entries", 100, -0.15, 0.15);
  tk_BS_dxy_hist = ibook.book1D("tk_BS_dxy", "Track d_{xy} w.r.t. BeamSpot;dxy_{BS} (cm);Entries", 100, -0.5, 0.5);
  tk_BS_dz_hist = ibook.book1D("tk_BS_dz", "Track d_{z} w.r.t. BeamSpot;dz_{BS} (cm);Entries", 100, -20.0, 20.0);

  // book the calo rechits histograms
  const std::array<std::string, 2> caloLabels = {{"Accepted", "Rejected"}};
  const std::array<std::string, 2> caloSuffixes = {{"", "_bad"}};
  for (int i = 0; i < 2; ++i) {
    ibook.setCurrentFolder(topfoldername_ + "/CaloRecHits" + caloLabels[i]);

    const std::string& lbl = caloLabels[i];
    const std::string& sfx = caloSuffixes[i];

    ebRecHitsNumber_hist[i] = ibook.book1D(
        "ebRechitsN" + sfx, "Number of EB RecHits (" + lbl + "); number of EB recHits; Entries", 100, 0.0, 1000.0);

    ebRecHits_energy_hist[i] =
        ibook.book1D("ebRechits_energy" + sfx,
                     "Energy spectrum of EB RecHits (" + lbl + "); Energy of EB recHits (Gev); Entries",
                     100,
                     0.0,
                     500.0);

    ebRecHits_time_hist[i] = ibook.book1D("ebRechits_time" + sfx,
                                          "Time of EB RecHits (" + lbl + "); Energy of EB recHits (ns); Entries",
                                          200,
                                          -100.,
                                          100.0);
    eeRecHitsNumber_hist[i] = ibook.book1D(
        "eeRechitsN" + sfx, "Number of EE RecHits (" + lbl + "); number of EE recHits; Entries", 100, 0.0, 1000.0);
    eeRecHits_energy_hist[i] =
        ibook.book1D("eeRechits_energy" + sfx,
                     "Energy spectrum of EE RecHits (" + lbl + "); Energy of EE recHits (GeV); Entries",
                     100,
                     0.0,
                     1000.0);
    eeRecHits_time_hist[i] = ibook.book1D("eeRechits_time" + sfx,
                                          "Time of EE RecHits (" + lbl + "); Time of EE recHits (ns); Entries",
                                          200,
                                          -100.0,
                                          100.0);

    ebRecHitsEtaPhiMap[i] = ibook.book2D("ebRecHitsEtaPhitMap" + sfx,
                                         "Occupancy map of EB rechits (" + lbl + ");ieta;iphi;Entries",
                                         171,
                                         -85.5,
                                         85.5,
                                         361,
                                         0.,
                                         361);

    ebRecHitsEtaPhiMap[i]->setOption("colz");

    eePlusRecHitsXYMap[i] = ibook.book2D("eePlusRecHitsEtaPhitMap" + sfx,
                                         "Occupancy map of EE+ rechits (" + lbl + ");ix;iy;Entries",
                                         100,
                                         1,
                                         101,
                                         100,
                                         1,
                                         101);

    eePlusRecHitsXYMap[i]->setOption("colz");

    eeMinusRecHitsXYMap[i] = ibook.book2D("eeMinusRecHitsEtaPhitMap" + sfx,
                                          "Occupancy map of EE- rechits (" + lbl + ");ix;iy;Entries",
                                          100,
                                          1,
                                          101,
                                          100,
                                          1,
                                          101);

    eeMinusRecHitsXYMap[i]->setOption("colz");
  }

  ibook.setCurrentFolder(topfoldername_ + "/CaloRecHitsAll");

  // now do HCAL
  const std::array<std::string, 3> subdets = {{"HBHE", "HB", "HE"}};

  // helper lambda
  auto toLower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
  };

  for (int i = 0; i < 3; ++i) {
    const std::string& subdet = subdets[i];
    std::string name = toLower(subdet);

    hbheRecHitsNumber_hist[i] =
        ibook.book1D(name + "RechitsN",
                     "number of " + subdet + " RecHits; Number of " + subdet + " recHits; RecHits",
                     100,
                     0.0,
                     2000.0);

    hbheRecHits_energy_hist[i] =
        ibook.book1D(name + "Rechits_energy",
                     "Energy spectrum of " + subdet + " RecHits; Energy of " + subdet + " recHits (GeV); RecHits",
                     100,
                     0.0,
                     200.0);

    // Energy > 5 GeV histograms
    hbheRecHits_energy_egt5_hist[i] = ibook.book1D(
        name + "StiffRechits_energy",
        "Energy spectrum of " + subdet + " RecHits  (E > 5 GeV);Energy of stiff " + subdet + " recHits (GeV); RecHits",
        100,
        0.0,
        30.0);

    hbheRecHits_time_hist[i] =
        ibook.book1D(name + "Rechits_time",
                     "Time of " + subdet + " RecHits; Time of " + subdet + " recHits (ns); RecHits",
                     100,
                     0.,
                     30.0);

    // Energy > 5 GeV histograms
    hbheRecHits_time_egt5_hist[i] =
        ibook.book1D(name + "StiffRechits_time",
                     "Time of " + subdet + " RecHits (E > 5 GeV); Time of stiff " + subdet + " recHits (ns); RecHits",
                     100,
                     0.,
                     30.0);
  }

  hbheRecHitsEtaPhiMap = ibook.book2D(
      "hbheRecHitsEtaPhitMap", "Occupancy map of HBHE rechits;ieta;iphi;RecHits", 61, -30.5, 30.5, 74, -0.5, 73.5);
  hbheRecHitsEtaPhiMap->setOption("colz");

  hbRecHitsEtaPhiMap = ibook.book2D(
      "hbRecHitsEtaPhitMap", "Occupancy map of HB rechits;ieta;iphi;RecHits", 83, -41.5, 41.5, 72, 0.5, 72.5);
  hbRecHitsEtaPhiMap->setOption("colz");

  heRecHitsEtaPhiMap = ibook.book2D(
      "heRecHitsEtaPhitMap", "Occupancy map of HE rechits;ieta;iphi;RecHits", 83, -41.5, 41.5, 72, 0.5, 72.5);
  heRecHitsEtaPhiMap->setOption("colz");
}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

void ScoutingCollectionMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("onlyScouting", false);
  desc.add<edm::InputTag>("electrons", edm::InputTag("hltScoutingEgammaPacker"));
  desc.add<edm::InputTag>("muons", edm::InputTag("hltScoutingMuonPackerNoVtx"));
  desc.add<edm::InputTag>("muonsVtx", edm::InputTag("hltScoutingMuonPackerVtx"));
  desc.add<edm::InputTag>("pfcands", edm::InputTag("hltScoutingPFPacker"));
  desc.add<edm::InputTag>("photons", edm::InputTag("hltScoutingEgammaPacker"));
  desc.add<edm::InputTag>("pfjets", edm::InputTag("hltScoutingPFPacker"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("hltScoutingTrackPacker"));
  desc.add<edm::InputTag>("displacedVertices", edm::InputTag("hltScoutingMuonPackerVtx", "displacedVtx"));
  desc.add<edm::InputTag>("displacedVerticesNoVtx", edm::InputTag("hltScoutingMuonPackerNoVtx", "displacedVtx"));
  desc.add<edm::InputTag>("primaryVertices", edm::InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"));
  desc.add<edm::InputTag>("pfMetPt", edm::InputTag("hltScoutingPFPacker", "pfMetPt"));
  desc.add<edm::InputTag>("pfMetPhi", edm::InputTag("hltScoutingPFPacker", "pfMetPhi"));
  desc.add<edm::InputTag>("rho", edm::InputTag("hltScoutingPFPacker", "rho"));
  desc.add<edm::InputTag>("onlineMetaDataDigis", edm::InputTag("onlineMetaDataDigis"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<edm::InputTag>("pfRecHitsEB", edm::InputTag("hltScoutingRecHitPacker", "EB"));
  desc.add<edm::InputTag>("pfRecHitsEE", edm::InputTag("hltScoutingRecHitPacker", "EE"));
  desc.add<edm::InputTag>("pfRecHitsHBHE", edm::InputTag("hltScoutingRecHitPacker", "HBHE"));
  desc.add<edm::InputTag>("pfCleanedRecHitsEB", edm::InputTag("hltScoutingRecHitPacker", "EBCleaned"));
  desc.add<edm::InputTag>("pfCleanedRecHitsEE", edm::InputTag("hltScoutingRecHitPacker", "EECleaned"));
  desc.add<std::string>("topfoldername", "HLT/ScoutingOffline/Miscellaneous");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ScoutingCollectionMonitor);
