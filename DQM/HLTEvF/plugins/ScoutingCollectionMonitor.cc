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
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// ROOT include files
#include <TMath.h>

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

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
  void setToken(edm::EDGetTokenT<T>& token, const edm::ParameterSet& iConfig, const std::string& name) {
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

  // Book a histogram for an integer-valued quantity in [nmin, nmax] with
  // one unit-width bin per integer, centred on the integer values.
  static dqm::reco::MonitorElement* bookIntHisto(
      DQMStore::IBooker& ibook, const std::string& name, const std::string& title, int nmin, int nmax) {
    return ibook.book1I(name, title, nmax - nmin + 1, nmin - 0.5, nmax + 0.5);
  }

  // Same as above, for a quantity that starts at 0 (multiplicities).
  static dqm::reco::MonitorElement* bookMultiplicity(DQMStore::IBooker& ibook,
                                                     const std::string& name,
                                                     const std::string& title,
                                                     int nmax) {
    return bookIntHisto(ibook, name, title, 0, nmax);
  }

  // Simple 3D point, used as a common reference point for impact parameter computations
  // (either the beam spot or a primary vertex).
  struct Point3D {
    float x;
    float y;
    float z;
  };

  // Impact parameters (dxy, dz) of a track w.r.t. a reference point, following the reco::TrackBase
  // convention: dxy(P) = (-(vx - Px) * py + (vy - Py) * px) / pt and
  //             dz(P)  = (vz - Pz) - ((vx - Px) * px + (vy - Py) * py) / pt * pz / pt
  static inline std::pair<float, float> trk_vtx_offSet(const Run3ScoutingTrack& tk, const Point3D& ref) {
    const auto pt = tk.tk_pt();
    const auto phi = tk.tk_phi();
    const auto eta = tk.tk_eta();

    const auto px = pt * std::cos(phi);
    const auto py = pt * std::sin(phi);
    const auto pz = pt * std::sinh(eta);
    const auto pt2 = pt * pt;

    const auto dx = tk.tk_vx() - ref.x;
    const auto dy = tk.tk_vy() - ref.y;
    const auto dz = tk.tk_vz() - ref.z;

    const auto tk_dxyPV = (-dx * py + dy * px) / pt;
    const auto tk_dzPV = dz - (dx * px + dy * py) * pz / pt2;

    return {tk_dxyPV, tk_dzPV};
  }

  // Upper edges of the multiplicity histograms.
  // They are configurable to accommodate the higher occupancies expected at Phase-2 (see fillDescriptions).
  struct MultiplicityRanges {
    int nTracks;
    int nPrimaryVertices;
    int nDisplacedVertices;
    int nMuons;
    int nElectrons;
    int nPhotons;
    int nPFJets;
    int nPFCands;
    int nEBRecHits;
    int nEERecHits;
    int nHBHERecHits;
    double pileUp;
  };

  // Binning (number of bins, lower and upper edge) of a 1D histogram, configurable from python
  struct Binning {
    int nbins;
    double min;
    double max;

    static Binning fromPSet(const edm::ParameterSet& pset) {
      return Binning{
          pset.getParameter<int>("nbins"), pset.getParameter<double>("min"), pset.getParameter<double>("max")};
    }
  };

  const bool onlyScouting_;
  const MultiplicityRanges ranges_;
  const Binning rhoBinning_;  // used for the rho histogram and for the rho axis of the rho vs PU profile
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

  // best electron tracks tokens
  // One ValueMap per track variable: each map is keyed on the electron
  // collection and already holds the best-track scalar for every electron.
  const edm::EDGetTokenT<edm::ValueMap<int>> vmBestTrackIndexToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> vmTrkd0Token_;
  const edm::EDGetTokenT<edm::ValueMap<float>> vmTrkdzToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> vmTrkptToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> vmTrketaToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> vmTrkphiToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> vmTrkpModeToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> vmTrketaModeToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> vmTrkphiModeToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> vmTrkqoverpModeErrorToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> vmTrkchi2overndfToken_;
  const edm::EDGetTokenT<edm::ValueMap<int>> vmTrkchargeToken_;

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
  dqm::reco::MonitorElement* nClusters_ele_hist;
  dqm::reco::MonitorElement* nCrystals_ele_hist;
  dqm::reco::MonitorElement* rechitZeroSuppression_ele_hist;
  dqm::reco::MonitorElement* nTracks_ele_hist;
  // ---- electron best track variables
  dqm::reco::MonitorElement* trkBestIdx_ele_hist;
  dqm::reco::MonitorElement* trkd0_ele_hist;
  dqm::reco::MonitorElement* trkdz_ele_hist;
  dqm::reco::MonitorElement* trkd0BS_ele_hist;
  dqm::reco::MonitorElement* trkdzBS_ele_hist;
  dqm::reco::MonitorElement* trkd0Vtx_ele_hist;
  dqm::reco::MonitorElement* trkdzVtx_ele_hist;
  dqm::reco::MonitorElement* trkpt_ele_hist;
  dqm::reco::MonitorElement* trketa_ele_hist;
  dqm::reco::MonitorElement* trkphi_ele_hist;
  dqm::reco::MonitorElement* trkpMode_ele_hist;
  dqm::reco::MonitorElement* trketaMode_ele_hist;
  dqm::reco::MonitorElement* trkphiMode_ele_hist;
  dqm::reco::MonitorElement* trkqoverpModeError_ele_hist;
  dqm::reco::MonitorElement* trkchi2overndf_ele_hist;
  dqm::reco::MonitorElement* trkcharge_ele_hist;

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
      ranges_([&iConfig]() {
        const auto& pset = iConfig.getParameter<edm::ParameterSet>("multiplicityRanges");
        return MultiplicityRanges{pset.getParameter<int>("nTracks"),
                                  pset.getParameter<int>("nPrimaryVertices"),
                                  pset.getParameter<int>("nDisplacedVertices"),
                                  pset.getParameter<int>("nMuons"),
                                  pset.getParameter<int>("nElectrons"),
                                  pset.getParameter<int>("nPhotons"),
                                  pset.getParameter<int>("nPFJets"),
                                  pset.getParameter<int>("nPFCands"),
                                  pset.getParameter<int>("nEBRecHits"),
                                  pset.getParameter<int>("nEERecHits"),
                                  pset.getParameter<int>("nHBHERecHits"),
                                  pset.getParameter<double>("pileUp")};
      }()),
      rhoBinning_(Binning::fromPSet(iConfig.getParameter<edm::ParameterSet>("rhoBinning"))),
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
      // ---- ValueMap tokens: instanceLabel must match what the producer puts ----
      vmBestTrackIndexToken_(consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("vmBestTrackIndex"))),
      vmTrkd0Token_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("vmTrkd0"))),
      vmTrkdzToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("vmTrkdz"))),
      vmTrkptToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("vmTrkpt"))),
      vmTrketaToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("vmTrketa"))),
      vmTrkphiToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("vmTrkphi"))),
      vmTrkpModeToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("vmTrkpMode"))),
      vmTrketaModeToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("vmTrketaMode"))),
      vmTrkphiModeToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("vmTrkphiMode"))),
      vmTrkqoverpModeErrorToken_(
          consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("vmTrkqoverpModeError"))),
      vmTrkchi2overndfToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("vmTrkchi2overndf"))),
      vmTrkchargeToken_(consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("vmTrkcharge"))),
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

  // get pile up (only available when running on the full HLT output, not on scouting-only data)
  if (!onlyScouting_) {
    if (!getValidHandle(iEvent, onlineMetaDataDigisToken_, onlineMetaDataDigisHandle, "avgPileUp")) {
      return;
    }
    const float avgPileUp = onlineMetaDataDigisHandle->avgPileUp();
    rhovsPU_hist->Fill(avgPileUp, *rhoH);
    PVvsPU_hist->Fill(avgPileUp, primaryVerticesH->size());
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
  // pdgId convention for the HF candidates follows reco::PFCandidate: 1 = HF hadron, 2 = HF e/gamma
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
    nClusters_pho_hist->Fill(pho.nClusters());
    nCrystals_pho_hist->Fill(pho.nCrystals());
    rechitZeroSuppression_pho_hist->Fill(pho.rechitZeroSuppression() ? -1. : 1.);
  }

  // determine the beamspot position (if it exists in the event)
  std::optional<Point3D> beamspotPosition;
  edm::Handle<reco::BeamSpot> beamSpotH;
  if (getValidHandle(iEvent, beamSpotToken_, beamSpotH, "beamSpot")) {
    beamspotPosition = Point3D{
        static_cast<float>(beamSpotH->x0()), static_cast<float>(beamSpotH->y0()), static_cast<float>(beamSpotH->z0())};
  }

  // lambda to find the primary vertex closest in z to a given longitudinal position
  auto findClosestVtx = [&](float dz0) -> const Run3ScoutingVertex* {
    const Run3ScoutingVertex* bestVtx = nullptr;
    float bestDist = std::numeric_limits<float>::max();

    for (const auto& vtx : *primaryVerticesH) {
      const float dist = std::abs(dz0 - vtx.z());

      if (dist < bestDist) {
        bestDist = dist;
        bestVtx = &vtx;
      }
    }

    return bestVtx;
  };

  // --- best electron track ValueMaps ---
  // These are produced by a separate module (Run3ScoutingElectronBestTrackProducer) and might not be
  // available in every workflow: if they are missing, skip only the best-track plots and keep filling
  // everything else.
  edm::Handle<edm::ValueMap<int>> vmBestIdxH;
  edm::Handle<edm::ValueMap<float>> vmD0H;
  edm::Handle<edm::ValueMap<float>> vmDzH;
  edm::Handle<edm::ValueMap<float>> vmPtH;
  edm::Handle<edm::ValueMap<float>> vmEtaH;
  edm::Handle<edm::ValueMap<float>> vmPhiH;
  edm::Handle<edm::ValueMap<float>> vmPModeH;
  edm::Handle<edm::ValueMap<float>> vmEtaModeH;
  edm::Handle<edm::ValueMap<float>> vmPhiModeH;
  edm::Handle<edm::ValueMap<float>> vmQoverpModeErrH;
  edm::Handle<edm::ValueMap<float>> vmChi2H;
  edm::Handle<edm::ValueMap<int>> vmChargeH;

  const bool haveBestTrackMaps =
      getValidHandle(iEvent, vmBestTrackIndexToken_, vmBestIdxH, "vmBestTrackIndex") &&
      getValidHandle(iEvent, vmTrkd0Token_, vmD0H, "vmTrkd0") &&
      getValidHandle(iEvent, vmTrkdzToken_, vmDzH, "vmTrkdz") &&
      getValidHandle(iEvent, vmTrkptToken_, vmPtH, "vmTrkpt") &&
      getValidHandle(iEvent, vmTrketaToken_, vmEtaH, "vmTrketa") &&
      getValidHandle(iEvent, vmTrkphiToken_, vmPhiH, "vmTrkphi") &&
      getValidHandle(iEvent, vmTrkpModeToken_, vmPModeH, "vmTrkpMode") &&
      getValidHandle(iEvent, vmTrketaModeToken_, vmEtaModeH, "vmTrketaMode") &&
      getValidHandle(iEvent, vmTrkphiModeToken_, vmPhiModeH, "vmTrkphiMode") &&
      getValidHandle(iEvent, vmTrkqoverpModeErrorToken_, vmQoverpModeErrH, "vmTrkqoverpModeError") &&
      getValidHandle(iEvent, vmTrkchi2overndfToken_, vmChi2H, "vmTrkchi2overndf") &&
      getValidHandle(iEvent, vmTrkchargeToken_, vmChargeH, "vmTrkcharge");

  // fill all the electron histograms
  for (std::size_t iEl = 0; iEl < electronsH->size(); ++iEl) {
    // Ref needed to index into the ValueMaps
    const edm::Ref<Run3ScoutingElectronCollection> elRef(electronsH, iEl);
    const Run3ScoutingElectron& ele = *elRef;

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
    nClusters_ele_hist->Fill(ele.nClusters());
    nCrystals_ele_hist->Fill(ele.nCrystals());
    rechitZeroSuppression_ele_hist->Fill(ele.rechitZeroSuppression() ? -1. : 1.);

    // ----- Track-vector size -----
    nTracks_ele_hist->Fill(static_cast<double>(ele.trkpt().size()));

    if (!haveBestTrackMaps)
      continue;

    // ----- Best-track scalars from ValueMaps -----
    // The producer sets the value to numeric_limits<float>::max() when no
    // best track was found (index == -1), so guard before filling.
    const int bestIdx = (*vmBestIdxH)[elRef];
    trkBestIdx_ele_hist->Fill(bestIdx);

    if (bestIdx < 0)
      continue;  // no valid track for this electron

    // All float maps are safe to fill: the producer guarantees they hold the
    // best-track value whenever bestIdx >= 0.
    const float d0 = (*vmD0H)[elRef];
    const float dz0 = (*vmDzH)[elRef];
    const float pt = (*vmPtH)[elRef];
    const float eta = (*vmEtaH)[elRef];
    const float phi = (*vmPhiH)[elRef];

    trkd0_ele_hist->Fill(d0);
    trkdz_ele_hist->Fill(dz0);

    // computations to get IP w.r.t. a point
    const float px = pt * std::cos(phi);
    const float py = pt * std::sin(phi);
    const float pz = pt * std::sinh(eta);
    const float pt2 = pt * pt;

    // lambda to compute the IP w.r.t. any reference point.
    // NB: the packer stores reco::Track::d0() = -dxy(), hence the sign of the transverse term
    // is opposite to the one used in trk_vtx_offSet for the tracks.
    auto computeIP = [&](const Point3D& ref) {
      const float d0_ref = d0 + (-ref.x * py + ref.y * px) / pt;
      const float dz_ref = dz0 - ref.z + (ref.x * px + ref.y * py) * pz / pt2;
      return std::pair<float, float>{d0_ref, dz_ref};
    };

    // compute w.r.t. beamspot (skip beamspot-based plots if not available)
    if (beamspotPosition) {
      const auto [d0_bs, dz_bs] = computeIP(*beamspotPosition);
      trkd0BS_ele_hist->Fill(d0_bs);
      trkdzBS_ele_hist->Fill(dz_bs);
    }

    // compute w.r.t. the closest primary vertex (skip if there are no primary vertices)
    if (const auto* vtx = findClosestVtx(dz0)) {
      const auto [d0_vtx, dz_vtx] = computeIP(Point3D{vtx->x(), vtx->y(), vtx->z()});
      trkd0Vtx_ele_hist->Fill(d0_vtx);
      trkdzVtx_ele_hist->Fill(dz_vtx);
    }

    trkpt_ele_hist->Fill(pt);
    trketa_ele_hist->Fill(eta);
    trkphi_ele_hist->Fill(phi);
    trkpMode_ele_hist->Fill((*vmPModeH)[elRef]);
    trketaMode_ele_hist->Fill((*vmEtaModeH)[elRef]);
    trkphiMode_ele_hist->Fill((*vmPhiModeH)[elRef]);
    trkqoverpModeError_ele_hist->Fill((*vmQoverpModeErrH)[elRef]);
    trkchi2overndf_ele_hist->Fill((*vmChi2H)[elRef]);
    trkcharge_ele_hist->Fill((*vmChargeH)[elRef]);
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
    if (tk.tk_ndof() > 0) {
      tk_chi2_ndof_tk_hist->Fill(tk.tk_chi2() / tk.tk_ndof());
      tk_chi2_prob_hist->Fill(TMath::Prob(tk.tk_chi2(), static_cast<int>(tk.tk_ndof())));
    }

    // loop on all the primary vertices and pick the one the track is closest to in dz
    std::optional<std::pair<float, float>> best_offset;
    for (const auto& vtx : *primaryVerticesH) {
      const auto offset = trk_vtx_offSet(tk, Point3D{vtx.x(), vtx.y(), vtx.z()});
      if (!best_offset || std::abs(offset.second) < std::abs(best_offset->second)) {
        best_offset = offset;
      }
    }

    // skip PV-based plots if there are no primary vertices in the event
    if (best_offset) {
      tk_PV_dxy_hist->Fill(best_offset->first);
      tk_PV_dz_hist->Fill(best_offset->second);
    }

    // skip beamspot-based plots if not available
    if (beamspotPosition) {
      const auto bs_offset = trk_vtx_offSet(tk, *beamspotPosition);
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

  // process the HBHE rechits
  edm::Handle<Run3ScoutingHBHERecHitCollection> hbheRecHitsH;
  if (!hbheRecHitsToken_.isUninitialized() &&
      getValidHandle(iEvent, hbheRecHitsToken_, hbheRecHitsH, "pfRecHitsHBHE")) {
    // energy threshold used to define the "stiff" rechits
    constexpr float kStiffRecHitEnergy = 5.f;  // GeV

    // index of the subdetector-specific MEs (see the ordering used in bookHistograms: HBHE, HB, HE)
    constexpr int kHBHE = 0;
    constexpr int kHB = 1;
    constexpr int kHE = 2;

    // counter of rechits per subdetector
    std::array<unsigned int, 3> nRecHits{{0, 0, 0}};

    auto fillHcalHistograms = [&](int index, const Run3ScoutingHBHERecHit& hit) {
      nRecHits[index]++;
      hbheRecHits_energy_hist[index]->Fill(hit.energy());
      hbheRecHits_time_hist[index]->Fill(hit.time());
      if (hit.energy() > kStiffRecHitEnergy) {
        hbheRecHits_energy_egt5_hist[index]->Fill(hit.energy());
        hbheRecHits_time_egt5_hist[index]->Fill(hit.time());
      }
    };

    for (const auto& hbheRecHit : *hbheRecHitsH) {
      const HcalDetId hcalid(hbheRecHit.detId());

      fillHcalHistograms(kHBHE, hbheRecHit);
      hbheRecHitsEtaPhiMap->Fill(hcalid.ieta(), hcalid.iphi());

      switch (hcalid.subdet()) {
        case HcalBarrel:
          fillHcalHistograms(kHB, hbheRecHit);
          hbRecHitsEtaPhiMap->Fill(hcalid.ieta(), hcalid.iphi());
          break;
        case HcalEndcap:
          fillHcalHistograms(kHE, hbheRecHit);
          heRecHitsEtaPhiMap->Fill(hcalid.ieta(), hcalid.iphi());
          break;
        default:
          edm::LogWarning("ScoutingCollectionMonitor")
              << "Unexpected HCAL subdetector " << hcalid.subdet() << " in the HBHE scouting rechit collection";
          break;
      }
    }

    for (int i = 0; i < 3; ++i) {
      hbheRecHitsNumber_hist[i]->Fill(nRecHits[i]);
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void ScoutingCollectionMonitor::bookHistograms(DQMStore::IBooker& ibook,
                                               edm::Run const& run,
                                               edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(topfoldername_);

  // Book multiplicity histograms in the topfolder.
  // Integer-valued quantities get one unit-width bin per integer (centred on the integer);
  // the upper edges are configurable via the "multiplicityRanges" PSet.
  nTracks_hist = bookMultiplicity(ibook, "nTracks", "Number of Tracks;N_{tracks};Entries", ranges_.nTracks);
  nPrimaryVertices_hist = bookMultiplicity(
      ibook, "nPrimaryVertices", "Number of Primary Vertices;N_{PV};Entries", ranges_.nPrimaryVertices);
  nDisplacedVertices_hist = bookMultiplicity(
      ibook, "nDisplacedVertices", "Number of Displaced Vertices (Vtx);N_{DV};Entries", ranges_.nDisplacedVertices);
  nDisplacedVerticesNoVtx_hist = bookMultiplicity(ibook,
                                                  "nDisplacedVerticesNoVtx",
                                                  "Number of Displaced Vertices (NoVtx);N_{DV}^{NoVtx};Entries",
                                                  ranges_.nDisplacedVertices);
  nMuons_hist = bookMultiplicity(ibook, "nMuons", "Number of Muons (NoVtx);N_{muons};Entries", ranges_.nMuons);
  nMuonsVtx_hist =
      bookMultiplicity(ibook, "nMuonsVtx", "Number of Muons (Vtx);N_{muons}^{Vtx};Entries", ranges_.nMuons);
  nElectrons_hist = bookMultiplicity(ibook, "nElectrons", "Number of Electrons;N_{ele};Entries", ranges_.nElectrons);
  nPhotons_hist = bookMultiplicity(ibook, "nPhotons", "Number of Photons;N_{photon};Entries", ranges_.nPhotons);
  nPFJets_hist = bookMultiplicity(ibook, "nPFJets", "Number of PF Jets;N_{jet};Entries", ranges_.nPFJets);
  nPFCands_hist = bookMultiplicity(ibook, "nPFCands", "Number of PF Candidates;N_{pfcand};Entries", ranges_.nPFCands);

  rho_hist = ibook.book1D("rho", "#rho; #rho (GeV); Entries", rhoBinning_.nbins, rhoBinning_.min, rhoBinning_.max);
  pfMetPhi_hist =
      ibook.book1D("pfMetPhi", "PF MET #phi; #phi (rad); Entries", 100, -std::numbers::pi, std::numbers::pi);
  pfMetPt_hist = ibook.book1D("pfMetPt", "PF MET p_{T}; p_{T} (GeV); Entries", 100, 0.0, 250.0);

  if (!onlyScouting_) {
    // one bin per unit of pile-up; the y-range of the profiles must contain all the values to be averaged
    const int nPUBins = static_cast<int>(std::ceil(ranges_.pileUp));
    PVvsPU_hist = ibook.bookProfile("PVvsPU",
                                    "Number of primary vertices vs pile up; pile up; #LTN_{PV}#GT",
                                    nPUBins,
                                    0.,
                                    ranges_.pileUp,
                                    0.,
                                    ranges_.nPrimaryVertices + 0.5,
                                    "");
    rhovsPU_hist = ibook.bookProfile("rhovsPU",
                                     "#rho vs pile up; pile up; #LT#rho#GT (GeV)",
                                     nPUBins,
                                     0.,
                                     ranges_.pileUp,
                                     rhoBinning_.min,
                                     rhoBinning_.max,
                                     "");
  }

  ibook.setCurrentFolder(topfoldername_ + "/PFcand");
  PF_pT_211_hist = ibook.book1DD("pT_posHad", "PF h^{+} p_{T};p_{T} (GeV);Entries", 100, 0.0, 13.0);
  PF_pT_n211_hist = ibook.book1DD("pT_negHad", "PF h^{-} p_{T};p_{T} (GeV);Entries", 100, 0.0, 14.0);
  PF_pT_130_hist = ibook.book1DD("pT_neuHad", "PF h^{0} p_{T};p_{T} (GeV);Entries", 100, 0.0, 20.0);
  PF_pT_22_hist = ibook.book1DD("pT_gamma", "PF #gamma p_{T};p_{T} (GeV);Entries", 100, 0.0, 18.0);
  PF_pT_13_hist = ibook.book1DD("pT_mu_plus", "PF #mu^{+} p_{T};p_{T} (GeV);Entries", 100, 0.0, 80.0);
  PF_pT_n13_hist = ibook.book1DD("pT_mu_minus", "PF #mu^{-} p_{T};p_{T} (GeV);Entries", 100, 0.0, 80.0);
  PF_pT_1_hist = ibook.book1DD("pT_HF_had", "PF HF h p_{T};p_{T} (GeV);Entries", 100, 0.0, 4.5);
  PF_pT_2_hist = ibook.book1DD("pT_HF_eg", "PF HF e/#gamma p_{T};p_{T} (GeV);Entries", 100, 0.0, 6.0);

  PF_eta_211_hist = ibook.book1DD("eta_posHad", "PF h^{+} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_n211_hist = ibook.book1DD("eta_negHad", "PF h^{-} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_130_hist = ibook.book1DD("eta_neuHad", "PF h^{0} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_22_hist = ibook.book1DD("eta_gamma", "PF #gamma #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_13_hist = ibook.book1DD("eta_mu_plus", "PF #mu^{+} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_n13_hist = ibook.book1DD("eta_mu_minus", "PF #mu^{-} #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_1_hist = ibook.book1DD("eta_HF_had", "PF HF h #eta;#eta;Entries", 100, -5.0, 5.0);
  PF_eta_2_hist = ibook.book1DD("eta_HF_eg", "PF HF e/#gamma #eta;#eta;Entries", 100, -5.0, 5.0);

  PF_phi_211_hist = ibook.book1DD("phi_posHad", "PF h^{+} #phi;#phi (rad);Entries", 100, -3.2, 3.2);
  PF_phi_n211_hist = ibook.book1DD("phi_negHad", "PF h^{-} #phi;#phi (rad);Entries", 100, -3.2, 3.2);
  PF_phi_130_hist = ibook.book1DD("phi_neuHad", "PF h^{0} #phi;#phi (rad);Entries", 100, -3.2, 3.2);
  PF_phi_22_hist = ibook.book1DD("phi_gamma", "PF #gamma #phi;#phi (rad);Entries", 100, -3.2, 3.2);
  PF_phi_13_hist = ibook.book1DD("phi_mu_plus", "PF #mu^{+} #phi;#phi (rad);Entries", 100, -3.2, 3.2);
  PF_phi_n13_hist = ibook.book1DD("phi_mu_minus", "PF #mu^{-} #phi;#phi (rad);Entries", 100, -3.2, 3.2);
  PF_phi_1_hist = ibook.book1DD("phi_HF_had", "PF HF h #phi;#phi (rad);Entries", 100, -3.2, 3.2);
  PF_phi_2_hist = ibook.book1DD("phi_HF_eg", "PF HF e/#gamma #phi;#phi (rad);Entries", 100, -3.2, 3.2);

  PF_vertex_211_hist =
      ibook.book1DD("vertexIndex_posHad", "PF h^{+} Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_n211_hist =
      ibook.book1DD("vertexIndex_negHad", "PF h^{-} Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_130_hist =
      ibook.book1DD("vertexIndex_neuHad", "PF h^{0} Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_22_hist = ibook.book1DD("vertexIndex_gamma", "PF #gamma Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_13_hist =
      ibook.book1DD("vertexIndex_mu_plus", "PF #mu^{+} Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_n13_hist =
      ibook.book1DD("vertexIndex_mu_minus", "PF #mu^{-} Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_1_hist = ibook.book1DD("vertexIndex_HF_had", "PF HF h Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);
  PF_vertex_2_hist =
      ibook.book1DD("vertexIndex_HF_eg", "PF HF e/#gamma Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);

  // the following variables make sense only if there is a Track

  PF_normchi2_211_hist =
      ibook.book1DD("normchi2_posHad", "PF h^{+} Normalized #chi^{2};#chi^{2}/ndof;Entries", 100, 0.0, 10.0);
  PF_normchi2_n211_hist =
      ibook.book1DD("normchi2_negHad", "PF h^{-} Normalized #chi^{2};#chi^{2}/ndof;Entries", 100, 0.0, 10.0);
  PF_normchi2_13_hist =
      ibook.book1DD("normchi2_mu_plus", "PF #mu^{+} Normalized #chi^{2};#chi^{2}/ndof;Entries", 100, 0.0, 10.0);
  PF_normchi2_n13_hist =
      ibook.book1DD("normchi2_mu_minus", "PF #mu^{-} Normalized #chi^{2};#chi^{2}/ndof;Entries", 100, 0.0, 10.0);

  PF_dz_211_hist = ibook.book1DD("dz_posHad", "PF h^{+} d_{z};d_{z} (cm);Entries", 100, -1.0, 1.0);
  PF_dz_n211_hist = ibook.book1DD("dz_negHad", "PF h^{-} d_{z};d_{z} (cm);Entries", 100, -1.0, 1.0);
  PF_dz_13_hist = ibook.book1DD("dz_mu_plus", "PF #mu^{+} d_{z};d_{z} (cm);Entries", 100, -1.0, 1.0);
  PF_dz_n13_hist = ibook.book1DD("dz_mu_minus", "PF #mu^{-} d_{z};d_{z} (cm);Entries", 100, -1.0, 1.0);

  PF_dxy_211_hist = ibook.book1DD("dxy_posHad", "PF h^{+} d_{xy};d_{xy} (cm);Entries", 100, -0.5, 0.5);
  PF_dxy_n211_hist = ibook.book1DD("dxy_negHad", "PF h^{-} d_{xy};d_{xy} (cm);Entries", 100, -0.5, 0.5);
  PF_dxy_13_hist = ibook.book1DD("dxy_mu_plus", "PF #mu^{+} d_{xy};d_{xy} (cm);Entries", 100, -0.5, 0.5);
  PF_dxy_n13_hist = ibook.book1DD("dxy_mu_minus", "PF #mu^{-} d_{xy};d_{xy} (cm);Entries", 100, -0.5, 0.5);

  PF_dzsig_211_hist =
      ibook.book1DD("dzsig_posHad", "PF h^{+} d_{z} Significance;d_{z}/#sigma_{dz};Entries", 100, -10.0, 10.0);
  PF_dzsig_n211_hist =
      ibook.book1DD("dzsig_negHad", "PF h^{-} d_{z} Significance;d_{z}/#sigma_{dz};Entries", 100, -10.0, 10.0);
  PF_dzsig_13_hist =
      ibook.book1DD("dzsig_mu_plus", "PF #mu^{+} d_{z} Significance;d_{z}/#sigma_{dz};Entries", 100, -10.0, 10.0);
  PF_dzsig_n13_hist =
      ibook.book1DD("dzsig_mu_minus", "PF #mu^{-} d_{z} Significance;d_{z}/#sigma_{dz};Entries", 100, -10.0, 10.0);

  PF_dxysig_211_hist =
      ibook.book1DD("dxysig_posHad", "PF h^{+} d_{xy} Significance;d_{xy}/#sigma_{dxy};Entries", 100, -10.0, 10.0);
  PF_dxysig_n211_hist =
      ibook.book1DD("dxysig_negHad", "PF h^{-} d_{xy} Significance;d_{xy}/#sigma_{dxy};Entries", 100, -10.0, 10.0);
  PF_dxysig_13_hist =
      ibook.book1DD("dxysig_mu_plus", "PF #mu^{+} d_{xy} Significance;d_{xy}/#sigma_{dxy};Entries", 100, -10.0, 10.0);
  PF_dxysig_n13_hist =
      ibook.book1DD("dxysig_mu_minus", "PF #mu^{-} d_{xy} Significance;d_{xy}/#sigma_{dxy};Entries", 100, -10.0, 10.0);

  // These variables are actually the difference between the PF candidate reconstructed kinematics and its bestTrack ones.
  // This behaviour is governed by the "relativeTrackVars" parameter of HLTScoutingPFProducer
  // see https://github.com/cms-sw/cmssw/blob/master/HLTrigger/JetMET/plugins/HLTScoutingPFProducer.cc#L177-L185no

  PF_trk_pt_211_hist = ibook.book1DD(
      "trk_pt_posHad", "PF h^{+} #Delta p_{T}(Track - Cand);#Delta p_{T}(Track - Cand) (GeV);Entries", 100, -0.01, 0.01);
  PF_trk_pt_n211_hist = ibook.book1DD(
      "trk_pt_negHad", "PF h^{-} #Delta p_{T}(Track - Cand);#Delta p_{T}(Track - Cand) (GeV);Entries", 100, -0.01, 0.01);
  PF_trk_pt_13_hist = ibook.book1DD("trk_pt_mu_plus",
                                    "PF #mu^{+} #Delta p_{T}(Track - Cand);#Delta p_{T}(Track - Cand) (GeV);Entries",
                                    100,
                                    -0.01,
                                    0.01);
  PF_trk_pt_n13_hist = ibook.book1DD("trk_pt_mu_minus",
                                     "PF #mu^{-} #Delta p_{T}(Track - Cand);#Delta p_{T}(Track - Cand) (GeV);Entries",
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
      "trk_phi_posHad", "PF h^{+} #Delta #phi(Track - Cand);#Delta #phi(Track - Cand) (rad);Entries", 100, -0.01, 0.01);
  PF_trk_phi_n211_hist = ibook.book1DD(
      "trk_phi_negHad", "PF h^{-} #Delta #phi(Track - Cand);#Delta #phi(Track - Cand) (rad);Entries", 100, -0.01, 0.01);
  PF_trk_phi_13_hist = ibook.book1DD("trk_phi_mu_plus",
                                     "PF #mu^{+} #Delta #phi(Track - Cand);#Delta #phi(Track - Cand) (rad);Entries",
                                     100,
                                     -0.01,
                                     0.01);
  PF_trk_phi_n13_hist = ibook.book1DD("trk_phi_mu_minus",
                                      "PF #mu^{-} #Delta #phi(Track - Cand);#Delta #phi(Track - Cand) (rad);Entries",
                                      100,
                                      -0.01,
                                      0.01);

  ibook.setCurrentFolder(topfoldername_ + "/Photon");
  pt_pho_hist = ibook.book1DD("pt_pho", "Photon p_{T}; p_{T} (GeV); Entries", 100, 0.0, 100.0);
  eta_pho_hist = ibook.book1DD("eta_pho", "Photon #eta; #eta; Entries", 100, -2.7, 2.7);
  phi_pho_hist = ibook.book1DD("phi_pho", "Photon #phi; #phi (rad); Entries", 100, -std::numbers::pi, std::numbers::pi);
  rawEnergy_pho_hist = ibook.book1DD("rawEnergy_pho", "Raw Energy Photon; Energy (GeV); Entries", 100, 0.0, 250.0);
  preshowerEnergy_pho_hist =
      ibook.book1DD("preshowerEnergy_pho", "Preshower Energy Photon; Energy (GeV); Entries", 100, 0.0, 8.0);
  corrEcalEnergyError_pho_hist = ibook.book1DD(
      "corrEcalEnergyError_pho", "Corrected ECAL Energy Error Photon; Energy Error (GeV); Entries", 100, 0.0, 20.0);
  sigmaIetaIeta_pho_hist =
      ibook.book1DD("sigmaIetaIeta_pho", "#sigma_{i#eta i#eta} Photon; #sigma_{i#eta i#eta}; Entries", 100, 0.0, 0.5);
  hOverE_pho_hist = ibook.book1DD("hOverE_pho", "H/E Photon; H/E; Entries", 100, 0.0, 1.5);
  ecalIso_pho_hist = ibook.book1DD("ecalIso_pho", "ECAL Isolation Photon; Isolation (GeV); Entries", 100, 0.0, 100.0);
  hcalIso_pho_hist = ibook.book1DD("hcalIso_pho", "HCAL Isolation Photon; Isolation (GeV); Entries", 100, 0.0, 100.0);
  trackIso_pho_hist = ibook.book1DD("trackIso_pho", "Track Isolation Photon; Isolation (GeV); Entries", 100, 0.0, 0.05);
  r9_pho_hist = ibook.book1DD("r9_pho", "R9 Photon; R9; Entries", 100, 0.0, 5);
  sMin_pho_hist = ibook.book1DD("sMin_pho", "sMin Photon; sMin; Entries", 100, 0.0, 3);
  sMaj_pho_hist = ibook.book1DD("sMaj_pho", "sMaj Photon; sMaj; Entries", 100, 0.0, 3);
  nClusters_pho_hist = ibook.book1I("nClusters_pho", "number of Clusters Photon; n. Clusters; Entries", 20, -0.5, 19.5);
  nCrystals_pho_hist =
      ibook.book1I("nCrystals_pho", "number of Crystals Photon; n. Crystals; Entries", 100, -0.5, 99.5);
  rechitZeroSuppression_pho_hist = ibook.book1I(
      "rechitZS_pho", "recHit ZS Photon; recHit zero suppression (-1 = True, 1 = False); Entries", 3, -1.5, 1.5);

  ibook.setCurrentFolder(topfoldername_ + "/Electron");
  pt_ele_hist = ibook.book1DD("pt_ele", "Electron p_{T}; p_{T} (GeV); Entries", 100, 0.0, 100.0);
  eta_ele_hist = ibook.book1DD("eta_ele", "Electron #eta; #eta; Entries", 100, -2.7, 2.7);
  phi_ele_hist =
      ibook.book1DD("phi_ele", "Electron #phi; #phi (rad); Entries", 100, -std::numbers::pi, std::numbers::pi);
  rawEnergy_ele_hist = ibook.book1DD("rawEnergy_ele", "Raw Energy Electron; Energy (GeV); Entries", 100, 0.0, 250.0);
  preshowerEnergy_ele_hist =
      ibook.book1DD("preshowerEnergy_ele", "Preshower Energy Electron; Energy (GeV); Entries", 100, 0.0, 10.0);
  corrEcalEnergyError_ele_hist = ibook.book1DD(
      "corrEcalEnergyError_ele", "Corrected ECAL Energy Error Electron; Energy Error (GeV); Entries", 100, 0.0, 20.0);
  dEtaIn_ele_hist = ibook.book1DD("dEtaIn_ele", "#Delta#eta_{in} Electron; #Delta#eta_{in}; Entries", 100, -0.05, 0.05);
  dPhiIn_ele_hist =
      ibook.book1DD("dPhiIn_ele", "#Delta#phi_{in} Electron; #Delta#phi_{in} (rad); Entries", 100, -0.5, 0.5);
  sigmaIetaIeta_ele_hist = ibook.book1DD(
      "sigmaIetaIeta_ele", "#sigma_{i#eta i#eta} Electron; #sigma_{i#eta i#eta}; Entries", 100, 0.0, 0.05);
  hOverE_ele_hist = ibook.book1DD("hOverE_ele", "H/E Electron; H/E; Entries", 100, 0.0, 0.3);
  ooEMOop_ele_hist = ibook.book1DD("ooEMOop_ele", "1/E - 1/p Electron; 1/E - 1/p (GeV^{-1}); Entries", 100, -0.3, 0.3);
  missingHits_ele_hist =
      bookMultiplicity(ibook, "missingHits_ele", "Missing Hits Electron; N_{missing hits}; Entries", 5);
  trackfbrem_ele_hist = ibook.book1DD("trackfbrem_ele", "Track f_{brem} Electron; f_{brem}; Entries", 100, -1.5, 1.0);
  ecalIso_ele_hist = ibook.book1DD("ecalIso_ele", "ECAL Isolation Electron; Isolation (GeV); Entries", 100, 0.0, 70.0);
  hcalIso_ele_hist = ibook.book1DD("hcalIso_ele", "HCAL Isolation Electron; Isolation (GeV); Entries", 100, 0.0, 60.0);
  trackIso_ele_hist =
      ibook.book1DD("trackIso_ele", "Track Isolation Electron; Isolation (GeV); Entries", 100, 0.0, 0.05);
  r9_ele_hist = ibook.book1DD("r9_ele", "R9 Electron; R9; Entries", 100, 0.0, 5);
  sMin_ele_hist = ibook.book1DD("sMin_ele", "sMin Electron; sMin; Entries", 100, 0.0, 3);
  sMaj_ele_hist = ibook.book1DD("sMaj_ele", "sMaj Electron; sMaj; Entries", 100, 0.0, 3);
  nClusters_ele_hist =
      ibook.book1I("nClusters_ele", "number of Clusters Electron; n. Clusters; Entries", 20, -0.5, 19.5);
  nCrystals_ele_hist =
      ibook.book1I("nCrystals_ele", "number of Crystals Electron; n. Crystals; Entries", 100, -0.5, 99.5);
  rechitZeroSuppression_ele_hist = ibook.book1I(
      "rechitZS_ele", "recHit ZS Electron; recHit zero suppression (-1 = True, 1 = False); Entries", 3, -1.5, 1.5);
  nTracks_ele_hist =
      bookMultiplicity(ibook, "nTracksPerElectron", "Number of tracks per electron;N_{trk};Electrons", 19);

  // --- Best-track variables (from ValueMaps) ---
  // index -1 means that no best track was found for the electron
  trkBestIdx_ele_hist = bookIntHisto(ibook, "trkBestIdx", "Best-track index;Best-track index;Electrons", -1, 19);
  trkd0_ele_hist = ibook.book1DD("trkd0", "Best-track d_{0};d_{0} (cm);Electrons", 100, -0.5, 0.5);
  trkdz_ele_hist = ibook.book1DD("trkdz", "Best-track d_{z};d_{z} (cm);Electrons", 100, -25, 25);
  trkd0BS_ele_hist = ibook.book1DD("trkd0BS", "Best-track d_{0}(BS);d_{0}(BS) (cm);Electrons", 100, -0.5, 0.5);
  trkdzBS_ele_hist = ibook.book1DD("trkdzBS", "Best-track d_{z}(BS);d_{z}(BS) (cm);Electrons", 100, -25, 25);
  trkd0Vtx_ele_hist = ibook.book1DD("trkd0Vtx", "Best-track d_{0}(PV);d_{0}(PV) (cm);Electrons", 100, -0.5, 0.5);
  trkdzVtx_ele_hist = ibook.book1DD("trkdzVtx", "Best-track d_{z}(PV);d_{z}(PV) (cm);Electrons", 100, -25, 25);
  trkpt_ele_hist = ibook.book1DD("trkpt", "Best-track p_{T};p_{T} (GeV);Electrons", 100, 0, 200);
  trketa_ele_hist = ibook.book1DD("trketa", "Best-track #eta;#eta;Electrons", 60, -3, 3);
  trkphi_ele_hist = ibook.book1DD("trkphi", "Best-track #phi;#phi (rad);Electrons", 64, -3.2, 3.2);
  trkpMode_ele_hist = ibook.book1DD("trkpMode", "Best-track p (mode);p_{mode} (GeV);Electrons", 100, 0, 200);
  trketaMode_ele_hist = ibook.book1DD("trketaMode", "Best-track #eta (mode);#eta_{mode};Electrons", 60, -3, 3);
  trkphiMode_ele_hist =
      ibook.book1DD("trkphiMode", "Best-track #phi (mode);#phi_{mode} (rad);Electrons", 64, -3.2, 3.2);
  trkqoverpModeError_ele_hist = ibook.book1DD(
      "trkqoverpModeError", "Best-track #sigma(q/p) (mode);#sigma(q/p)_{mode} (GeV^{-1});Electrons", 100, 0, 0.01);
  trkchi2overndf_ele_hist =
      ibook.book1DD("trkchi2overndf", "Best-track #chi^{2}/ndof;#chi^{2}/ndof;Electrons", 100, 0, 10);
  trkcharge_ele_hist = bookIntHisto(ibook, "trkcharge", "Best-track charge;charge;Electrons", -1, 1);

  // book the muon histograms (noVtx and Vtx collections)
  const std::array<std::string, 2> muonLabels = {{"muonsNoVtx", "muonsVtx"}};
  const std::array<std::string, 2> suffixes = {{"_noVtx", "_Vtx"}};
  for (int i = 0; i < 2; ++i) {
    ibook.setCurrentFolder(topfoldername_ + "/" + muonLabels[i]);

    const std::string& sfx = suffixes[i];
    const std::string& lbl = muonLabels[i];

    pt_mu_hist[i] = ibook.book1DD("pt_mu" + sfx, "Muon p_{T} (" + lbl + "); p_{T} (GeV); Entries", 100, 0.0, 200.0);
    eta_mu_hist[i] = ibook.book1DD("eta_mu" + sfx, "Muon #eta (" + lbl + "); #eta; Entries", 100, -2.7, 2.7);
    phi_mu_hist[i] = ibook.book1DD(
        "phi_mu" + sfx, "Muon #phi (" + lbl + "); #phi (rad); Entries", 100, -std::numbers::pi, std::numbers::pi);
    // the muon type is a bit mask (see reco::Muon::MuonType), so it is not a simple counter
    type_mu_hist[i] = bookMultiplicity(ibook, "type_mu" + sfx, "Muon Type (" + lbl + "); Type (bit mask); Entries", 31);
    charge_mu_hist[i] = bookIntHisto(ibook, "charge_mu" + sfx, "Muon Charge (" + lbl + "); Charge; Entries", -1, 1);
    normalizedChi2_mu_hist[i] = ibook.book1DD(
        "normalizedChi2_mu" + sfx, "Normalized #chi^{2} (" + lbl + "); #chi^{2}/ndof; Entries", 100, 0.0, 10.0);
    ecalIso_mu_hist[i] = ibook.book1DD(
        "ecalIso_mu" + sfx, "ECAL Isolation Muon (" + lbl + "); Isolation (GeV); Entries", 100, 0.0, 100.0);
    hcalIso_mu_hist[i] = ibook.book1DD(
        "hcalIso_mu" + sfx, "HCAL Isolation Muon (" + lbl + "); Isolation (GeV); Entries", 100, 0.0, 100.0);
    trackIso_mu_hist[i] = ibook.book1DD(
        "trackIso_mu" + sfx, "Track Isolation Muon (" + lbl + "); Isolation (GeV); Entries", 100, 0.0, 10.0);
    nValidStandAloneMuonHits_mu_hist[i] = bookMultiplicity(
        ibook, "nValidStandAloneMuonHits_mu" + sfx, "Valid Standalone Muon Hits (" + lbl + "); Hits; Entries", 50);
    nStandAloneMuonMatchedStations_mu_hist[i] =
        bookMultiplicity(ibook,
                         "nStandAloneMuonMatchedStations_mu" + sfx,
                         "Standalone Muon Matched Stations (" + lbl + "); Stations; Entries",
                         10);
    nValidRecoMuonHits_mu_hist[i] =
        bookMultiplicity(ibook, "nValidRecoMuonHits_mu" + sfx, "Valid Reco Muon Hits (" + lbl + "); Hits; Entries", 50);
    nRecoMuonChambers_mu_hist[i] = bookMultiplicity(
        ibook, "nRecoMuonChambers_mu" + sfx, "Reco Muon Chambers (" + lbl + "); Chambers; Entries", 20);
    nRecoMuonChambersCSCorDT_mu_hist[i] =
        bookMultiplicity(ibook,
                         "nRecoMuonChambersCSCorDT_mu" + sfx,
                         "Reco Muon Chambers (CSC or DT) (" + lbl + "); Chambers; Entries",
                         14);
    nRecoMuonMatches_mu_hist[i] =
        bookMultiplicity(ibook, "nRecoMuonMatches_mu" + sfx, "Reco Muon Matches (" + lbl + "); Matches; Entries", 10);
    nRecoMuonMatchedStations_mu_hist[i] = bookMultiplicity(
        ibook, "nRecoMuonMatchedStations_mu" + sfx, "Reco Muon Matched Stations (" + lbl + "); Stations; Entries", 10);
    nRecoMuonExpectedMatchedStations_mu_hist[i] =
        bookMultiplicity(ibook,
                         "nRecoMuonExpectedMatchedStations_mu" + sfx,
                         "Reco Muon Expected Matched Stations (" + lbl + "); Stations; Entries",
                         10);
    // 4 stations -> 4-bit mask
    recoMuonStationMask_mu_hist[i] = bookMultiplicity(
        ibook, "recoMuonStationMask_mu" + sfx, "Reco Muon Station Mask (" + lbl + "); Mask (bits); Entries", 15);
    nRecoMuonMatchedRPCLayers_mu_hist[i] = bookMultiplicity(
        ibook, "nRecoMuonMatchedRPCLayers_mu" + sfx, "Reco Muon Matched RPC Layers (" + lbl + "); Layers; Entries", 6);
    // 6 RPC layers -> 6-bit mask
    recoMuonRPClayerMask_mu_hist[i] = bookMultiplicity(
        ibook, "recoMuonRPClayerMask_mu" + sfx, "Reco Muon RPC Layer Mask (" + lbl + "); Mask (bits); Entries", 63);
    nValidPixelHits_mu_hist[i] =
        bookMultiplicity(ibook, "nValidPixelHits_mu" + sfx, "Valid Pixel Hits (" + lbl + "); Hits; Entries", 20);
    nValidStripHits_mu_hist[i] =
        bookMultiplicity(ibook, "nValidStripHits_mu" + sfx, "Valid Strip Hits (" + lbl + "); Hits; Entries", 50);
    nPixelLayersWithMeasurement_mu_hist[i] =
        bookMultiplicity(ibook,
                         "nPixelLayersWithMeasurement_mu" + sfx,
                         "Pixel Layers with Measurement (" + lbl + "); Layers; Entries",
                         10);
    nTrackerLayersWithMeasurement_mu_hist[i] =
        bookMultiplicity(ibook,
                         "nTrackerLayersWithMeasurement_mu" + sfx,
                         "Tracker Layers with Measurement (" + lbl + "); Layers; Entries",
                         20);
    trk_chi2_mu_hist[i] =
        ibook.book1DD("trk_chi2_mu" + sfx, "Muon Tracker #chi^{2} (" + lbl + "); #chi^{2}; Entries", 100, 0.0, 100.0);
    trk_ndof_mu_hist[i] =
        bookMultiplicity(ibook, "trk_ndof_mu" + sfx, "Muon Tracker ndof (" + lbl + "); ndof; Entries", 100);
    trk_dxy_mu_hist[i] =
        ibook.book1DD("trk_dxy_mu" + sfx, "Muon Tracker d_{xy} (" + lbl + "); d_{xy} (cm); Entries", 100, -0.5, 0.5);
    trk_dz_mu_hist[i] =
        ibook.book1DD("trk_dz_mu" + sfx, "Muon Tracker d_{z} (" + lbl + "); d_{z} (cm); Entries", 100, -20.0, 20.0);
    trk_qoverp_mu_hist[i] =
        ibook.book1DD("trk_qoverp_mu" + sfx, "Muon q/p (" + lbl + "); q/p (GeV^{-1}); Entries", 100, -1, 1);
    trk_lambda_mu_hist[i] =
        ibook.book1DD("trk_lambda_mu" + sfx, "Muon #lambda (" + lbl + "); #lambda (rad); Entries", 100, -2, 2);
    trk_pt_mu_hist[i] =
        ibook.book1DD("trk_pt_mu" + sfx, "Muon Tracker p_{T} (" + lbl + "); p_{T} (GeV); Entries", 100, 0.0, 200.0);
    trk_phi_mu_hist[i] = ibook.book1DD("trk_phi_mu" + sfx,
                                       "Muon Tracker #phi (" + lbl + "); #phi (rad); Entries",
                                       100,
                                       -std::numbers::pi,
                                       std::numbers::pi);
    trk_eta_mu_hist[i] =
        ibook.book1DD("trk_eta_mu" + sfx, "Muon Tracker #eta (" + lbl + "); #eta; Entries", 100, -3.0, 3.0);
    trk_dxyError_mu_hist[i] = ibook.book1DD(
        "trk_dxyError_mu" + sfx, "Muon d_{xy} Error (" + lbl + "); d_{xy} Error (cm); Entries", 100, 0.0, 0.05);
    trk_dzError_mu_hist[i] = ibook.book1DD(
        "trk_dzError_mu" + sfx, "Muon d_{z} Error (" + lbl + "); d_{z} Error (cm); Entries", 100, 0.0, 0.05);
    trk_qoverpError_mu_hist[i] = ibook.book1DD(
        "trk_qoverpError_mu" + sfx, "Muon q/p Error (" + lbl + "); q/p Error (GeV^{-1}); Entries", 100, 0.0, 0.01);
    trk_lambdaError_mu_hist[i] = ibook.book1DD(
        "trk_lambdaError_mu" + sfx, "Muon #lambda Error (" + lbl + "); #lambda Error (rad); Entries", 100, 0.0, 0.1);
    trk_phiError_mu_hist[i] = ibook.book1DD(
        "trk_phiError_mu" + sfx, "Muon #phi Error (" + lbl + "); #phi Error (rad); Entries", 100, 0.0, 0.01);
    trk_dsz_mu_hist[i] =
        ibook.book1DD("trk_dsz_mu" + sfx, "Muon d_{sz} (" + lbl + "); d_{sz} (cm); Entries", 100, -2, 2);
    trk_dszError_mu_hist[i] = ibook.book1DD(
        "trk_dszError_mu" + sfx, "Muon d_{sz} Error (" + lbl + "); d_{sz} Error (cm); Entries", 100, 0.0, 0.05);
    trk_qoverp_lambda_cov_mu_hist[i] =
        ibook.book1DD("trk_qoverp_lambda_cov_mu" + sfx,
                      "Muon q/p-#lambda Covariance (" + lbl + "); Cov(q/p, #lambda); Entries",
                      100,
                      -0.001,
                      0.001);
    trk_qoverp_phi_cov_mu_hist[i] = ibook.book1DD("trk_qoverp_phi_cov_mu" + sfx,
                                                  "Muon q/p-#phi Covariance (" + lbl + "); Cov(q/p, #phi); Entries",
                                                  100,
                                                  -0.001,
                                                  0.001);
    trk_qoverp_dxy_cov_mu_hist[i] = ibook.book1DD("trk_qoverp_dxy_cov_mu" + sfx,
                                                  "Muon q/p-d_{xy} Covariance (" + lbl + "); Cov(q/p, d_{xy}); Entries",
                                                  100,
                                                  -0.001,
                                                  0.001);
    trk_qoverp_dsz_cov_mu_hist[i] = ibook.book1DD("trk_qoverp_dsz_cov_mu" + sfx,
                                                  "Muon q/p-d_{sz} Covariance (" + lbl + "); Cov(q/p, d_{sz}); Entries",
                                                  100,
                                                  -0.001,
                                                  0.001);
    trk_lambda_phi_cov_mu_hist[i] =
        ibook.book1DD("trk_lambda_phi_cov_mu" + sfx,
                      "Muon #lambda-#phi Covariance (" + lbl + "); Cov(#lambda, #phi); Entries",
                      100,
                      -0.001,
                      0.001);
    trk_lambda_dxy_cov_mu_hist[i] =
        ibook.book1DD("trk_lambda_dxy_cov_mu" + sfx,
                      "Muon #lambda-d_{xy} Covariance (" + lbl + "); Cov(#lambda, d_{xy}); Entries",
                      100,
                      -0.001,
                      0.001);
    trk_lambda_dsz_cov_mu_hist[i] =
        ibook.book1DD("trk_lambda_dsz_cov_mu" + sfx,
                      "Muon #lambda-d_{sz} Covariance (" + lbl + "); Cov(#lambda, d_{sz}); Entries",
                      100,
                      -0.001,
                      0.001);
    trk_phi_dxy_cov_mu_hist[i] = ibook.book1DD("trk_phi_dxy_cov_mu" + sfx,
                                               "Muon #phi-d_{xy} Covariance (" + lbl + "); Cov(#phi, d_{xy}); Entries",
                                               100,
                                               -0.001,
                                               0.001);
    trk_phi_dsz_cov_mu_hist[i] = ibook.book1DD("trk_phi_dsz_cov_mu" + sfx,
                                               "Muon #phi-d_{sz} Covariance (" + lbl + "); Cov(#phi, d_{sz}); Entries",
                                               100,
                                               -0.001,
                                               0.001);
    trk_dxy_dsz_cov_mu_hist[i] =
        ibook.book1DD("trk_dxy_dsz_cov_mu" + sfx,
                      "Muon d_{xy}-d_{sz} Covariance (" + lbl + "); Cov(d_{xy}, d_{sz}); Entries",
                      100,
                      -0.001,
                      0.001);
    trk_vx_mu_hist[i] =
        ibook.book1DD("trk_vx_mu" + sfx, "Muon track reference point x (" + lbl + "); x (cm); Entries", 100, -0.5, 0.5);
    trk_vy_mu_hist[i] =
        ibook.book1DD("trk_vy_mu" + sfx, "Muon track reference point y (" + lbl + "); y (cm); Entries", 100, -0.5, 0.5);
    trk_vz_mu_hist[i] = ibook.book1DD(
        "trk_vz_mu" + sfx, "Muon track reference point z (" + lbl + "); z (cm); Entries", 100, -20.0, 20.0);
  }

  ibook.setCurrentFolder(topfoldername_ + "/PFJet");
  pt_pfj_hist = ibook.book1DD("pt_pfj", "PF Jet p_{T}; p_{T} (GeV); Entries", 100, 0.0, 150.0);
  eta_pfj_hist = ibook.book1DD("eta_pfj", "PF Jet #eta; #eta; Entries", 100, -5.0, 5.0);
  phi_pfj_hist = ibook.book1DD("phi_pfj", "PF Jet #phi; #phi (rad); Entries", 100, -std::numbers::pi, std::numbers::pi);
  m_pfj_hist = ibook.book1DD("m_pfj", "PF Jet Mass; Mass (GeV); Entries", 100, 0.0, 40.0);
  jetArea_pfj_hist =
      ibook.book1DD("jetArea_pfj", "PF Jet Area; Area (#Delta#eta #times #Delta#phi); Entries", 100, 0.0, 0.8);
  chargedHadronEnergy_pfj_hist =
      ibook.book1DD("chargedHadronEnergy_pfj", "PF Jet Charged Hadron Energy; Energy (GeV); Entries", 100, 0.0, 150.0);
  neutralHadronEnergy_pfj_hist =
      ibook.book1DD("neutralHadronEnergy_pfj", "PF Jet Neutral Hadron Energy; Energy (GeV); Entries", 100, 0.0, 600.0);
  photonEnergy_pfj_hist =
      ibook.book1DD("photonEnergy_pfj", "PF Jet Photon Energy; Energy (GeV); Entries", 100, 0.0, 90.0);
  electronEnergy_pfj_hist =
      ibook.book1DD("electronEnergy_pfj", "PF Jet Electron Energy; Energy (GeV); Entries", 100, 0.0, 3.0);
  muonEnergy_pfj_hist = ibook.book1DD("muonEnergy_pfj", "PF Jet Muon Energy; Energy (GeV); Entries", 100, 0.0, 3.0);
  HFHadronEnergy_pfj_hist =
      ibook.book1DD("HFHadronEnergy_pfj", "PF Jet HF Hadron Energy; Energy (GeV); Entries", 100, 0.0, 300.0);
  HFEMEnergy_pfj_hist = ibook.book1DD("HFEMEnergy_pfj", "PF Jet HF EM Energy; Energy (GeV); Entries", 100, 0.0, 300.0);
  chargedHadronMultiplicity_pfj_hist = bookMultiplicity(
      ibook, "chargedHadronMultiplicity_pfj", "PF Jet Charged Hadron Multiplicity; Multiplicity; Entries", 50);
  neutralHadronMultiplicity_pfj_hist = bookMultiplicity(
      ibook, "neutralHadronMultiplicity_pfj", "PF Jet Neutral Hadron Multiplicity; Multiplicity; Entries", 25);
  photonMultiplicity_pfj_hist =
      bookMultiplicity(ibook, "photonMultiplicity_pfj", "PF Jet Photon Multiplicity; Multiplicity; Entries", 50);
  electronMultiplicity_pfj_hist =
      bookMultiplicity(ibook, "electronMultiplicity_pfj", "PF Jet Electron Multiplicity; Multiplicity; Entries", 5);
  muonMultiplicity_pfj_hist =
      bookMultiplicity(ibook, "muonMultiplicity_pfj", "PF Jet Muon Multiplicity; Multiplicity; Entries", 5);
  HFHadronMultiplicity_pfj_hist =
      bookMultiplicity(ibook, "HFHadronMultiplicity_pfj", "PF Jet HF Hadron Multiplicity; Multiplicity; Entries", 20);
  HFEMMultiplicity_pfj_hist =
      bookMultiplicity(ibook, "HFEMMultiplicity_pfj", "PF Jet HF EM Multiplicity; Multiplicity; Entries", 20);
  HOEnergy_pfj_hist = ibook.book1DD("HOEnergy_pfj", "PF Jet HO Energy; Energy (GeV); Entries", 100, 0.0, 5.0);
  mvaDiscriminator_pfj_hist =
      ibook.book1DD("mvaDiscriminator_pfj", "PF Jet MVA Discriminator; Score; Entries", 100, -1.0, 1.0);

  ibook.setCurrentFolder(topfoldername_ + "/PrimaryVertex");
  x_pv_hist = ibook.book1DD("x_pv", "Primary Vertex X Position; x (cm); Entries", 100, -0.5, 0.5);
  y_pv_hist = ibook.book1DD("y_pv", "Primary Vertex Y Position; y (cm); Entries", 100, -0.5, 0.5);
  z_pv_hist = ibook.book1DD("z_pv", "Primary Vertex Z Position; z (cm); Entries", 100, -20.0, 20.0);
  zError_pv_hist = ibook.book1DD("zError_pv", "Primary Vertex Z Error; z Error (cm); Entries", 100, 0.0, 0.05);
  xError_pv_hist = ibook.book1DD("xError_pv", "Primary Vertex X Error; x Error (cm); Entries", 100, 0.0, 0.05);
  yError_pv_hist = ibook.book1DD("yError_pv", "Primary Vertex Y Error; y Error (cm); Entries", 100, 0.0, 0.05);
  tracksSize_pv_hist =
      bookMultiplicity(ibook, "tracksSize_pv", "Number of Tracks at Primary Vertex; Tracks; Entries", 100);
  chi2_pv_hist = ibook.book1DD("chi2_pv", "Primary Vertex #chi^{2}; #chi^{2}; Entries", 100, 0.0, 50.0);
  ndof_pv_hist = bookMultiplicity(ibook, "ndof_pv", "Primary Vertex ndof; ndof; Entries", 100);
  isValidVtx_pv_hist =
      bookIntHisto(ibook, "isValidVtx_pv", "Is Valid Primary Vertex?; 0 = False, 1 = True; Entries", 0, 1);
  xyCov_pv_hist =
      ibook.book1DD("xyCov_pv", "Primary Vertex XY Covariance; Cov(x,y) (cm^{2}); Entries", 100, -0.01, 0.01);
  xzCov_pv_hist =
      ibook.book1DD("xzCov_pv", "Primary Vertex XZ Covariance; Cov(x,z) (cm^{2}); Entries", 100, -0.01, 0.01);
  yzCov_pv_hist =
      ibook.book1DD("yzCov_pv", "Primary Vertex YZ Covariance; Cov(y,z) (cm^{2}); Entries", 100, -0.01, 0.01);

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
    tracksSize_vtx_hist[i] = bookMultiplicity(
        ibook, "tracksSize_vtx" + sfx, "Number of Tracks at Vertex (" + lbl + "); Tracks; Entries", 10);
    chi2_vtx_hist[i] =
        ibook.book1DD("chi2_vtx" + sfx, "Vertex #chi^{2} (" + lbl + "); #chi^{2}; Entries", 100, 0.0, 5.0);
    ndof_vtx_hist[i] = bookMultiplicity(ibook, "ndof_vtx" + sfx, "Vertex ndof (" + lbl + "); ndof; Entries", 10);
    isValidVtx_vtx_hist[i] = bookIntHisto(
        ibook, "isValidVtx_vtx" + sfx, "Is Valid Vertex? (" + lbl + "); 0 = False, 1 = True; Entries", 0, 1);
    xyCov_vtx_hist[i] = ibook.book1DD(
        "xyCov_vtx" + sfx, "Vertex XY Covariance (" + lbl + "); Cov(x,y) (cm^{2}); Entries", 100, -0.01, 0.01);
    xzCov_vtx_hist[i] = ibook.book1DD(
        "xzCov_vtx" + sfx, "Vertex XZ Covariance (" + lbl + "); Cov(x,z) (cm^{2}); Entries", 100, -0.01, 0.01);
    yzCov_vtx_hist[i] = ibook.book1DD(
        "yzCov_vtx" + sfx, "Vertex YZ Covariance (" + lbl + "); Cov(y,z) (cm^{2}); Entries", 100, -0.01, 0.01);
  }

  ibook.setCurrentFolder(topfoldername_ + "/Tracking");
  tk_pt_tk_hist = ibook.book1DD("tk_pt_tk", "Track p_{T}; p_{T} (GeV); Entries", 100, 0.0, 30.0);
  tk_eta_tk_hist = ibook.book1DD("tk_eta_tk", "Track #eta; #eta; Entries", 100, -2.7, 2.7);
  tk_phi_tk_hist =
      ibook.book1DD("tk_phi_tk", "Track #phi; #phi (rad); Entries", 100, -std::numbers::pi, std::numbers::pi);
  tk_chi2_tk_hist = ibook.book1DD("tk_chi2_tk", "Track #chi^{2}; #chi^{2}; Entries", 100, 0.0, 50.0);
  tk_ndof_tk_hist = bookMultiplicity(ibook, "tk_ndof_tk", "Track ndof; ndof; Entries", 30);
  tk_charge_tk_hist = bookIntHisto(ibook, "tk_charge_tk", "Track Charge; Charge; Entries", -1, 1);
  tk_dxy_tk_hist = ibook.book1DD("tk_dxy_tk", "Track d_{xy}; d_{xy} (cm); Entries", 100, -0.5, 0.5);
  tk_dz_tk_hist = ibook.book1DD("tk_dz_tk", "Track d_{z}; d_{z} (cm); Entries", 100, -20.0, 20.0);
  tk_nValidPixelHits_tk_hist =
      bookMultiplicity(ibook, "tk_nValidPixelHits_tk", "Track Valid Pixel Hits; Hits; Entries", 20);
  tk_nTrackerLayersWithMeasurement_tk_hist = bookMultiplicity(
      ibook, "tk_nTrackerLayersWithMeasurement_tk", "Track Tracker Layers with Measurement; Layers; Entries", 20);
  tk_nValidStripHits_tk_hist =
      bookMultiplicity(ibook, "tk_nValidStripHits_tk", "Track Valid Strip Hits; Hits; Entries", 50);
  tk_qoverp_tk_hist = ibook.book1DD("tk_qoverp_tk", "Track q/p; q/p (GeV^{-1}); Entries", 100, -1.0, 1.0);
  tk_lambda_tk_hist = ibook.book1DD("tk_lambda_tk", "Track #lambda; #lambda (rad); Entries", 100, -2, 2);
  tk_dxy_Error_tk_hist =
      ibook.book1DD("tk_dxy_Error_tk", "Track d_{xy} Error; d_{xy} Error (cm); Entries", 100, 0.0, 0.05);
  tk_dz_Error_tk_hist = ibook.book1DD("tk_dz_Error_tk", "Track d_{z} Error; d_{z} Error (cm); Entries", 100, 0.0, 0.05);
  tk_qoverp_Error_tk_hist =
      ibook.book1DD("tk_qoverp_Error_tk", "Track q/p Error; q/p Error (GeV^{-1}); Entries", 100, 0.0, 0.05);
  tk_lambda_Error_tk_hist =
      ibook.book1DD("tk_lambda_Error_tk", "Track #lambda Error; #lambda Error (rad); Entries", 100, 0.0, 0.1);
  tk_phi_Error_tk_hist =
      ibook.book1DD("tk_phi_Error_tk", "Track #phi Error; #phi Error (rad); Entries", 100, 0.0, 0.01);
  tk_dsz_tk_hist = ibook.book1DD("tk_dsz_tk", "Track d_{sz}; d_{sz} (cm); Entries", 100, -2, 2);
  tk_dsz_Error_tk_hist =
      ibook.book1DD("tk_dsz_Error_tk", "Track d_{sz} Error; d_{sz} Error (cm); Entries", 100, 0.0, 0.05);
  // index -1 means that the track is not associated to any primary vertex
  tk_vtxInd_tk_hist =
      bookIntHisto(ibook, "tk_vtxInd_tk", "Track Vertex Index; Vertex index; Entries", -1, ranges_.nPrimaryVertices);
  tk_vx_tk_hist = ibook.book1DD("tk_vx_tk", "Track reference point x; x (cm); Entries", 100, -0.5, 0.5);
  tk_vy_tk_hist = ibook.book1DD("tk_vy_tk", "Track reference point y; y (cm); Entries", 100, -0.5, 0.5);
  tk_vz_tk_hist = ibook.book1DD("tk_vz_tk", "Track reference point z; z (cm); Entries", 100, -20.0, 20.0);
  tk_chi2_ndof_tk_hist = ibook.book1DD("tk_chi2_ndof_tk", "Track Reduced #chi^{2}; #chi^{2}/ndof; Entries", 100, 0, 10);
  tk_chi2_prob_hist =
      ibook.book1DD("tk_chi2_prob_hist", "Track #chi^{2} probability; p(#chi^{2}, ndof); Entries", 100, 0, 1);
  tk_PV_dz_hist = ibook.book1DD("tk_PV_dz", "Track d_{z} w.r.t. PV; d_{z}(PV) (cm); Entries", 100, -0.35, 0.35);
  tk_PV_dxy_hist = ibook.book1DD("tk_PV_dxy", "Track d_{xy} w.r.t. PV; d_{xy}(PV) (cm); Entries", 100, -0.15, 0.15);
  tk_BS_dxy_hist = ibook.book1DD("tk_BS_dxy", "Track d_{xy} w.r.t. BeamSpot; d_{xy}(BS) (cm); Entries", 100, -0.5, 0.5);
  tk_BS_dz_hist = ibook.book1DD("tk_BS_dz", "Track d_{z} w.r.t. BeamSpot; d_{z}(BS) (cm); Entries", 100, -20.0, 20.0);

  // book the calo rechits histograms
  const std::array<std::string, 2> caloLabels = {{"Accepted", "Rejected"}};
  const std::array<std::string, 2> caloSuffixes = {{"", "_bad"}};
  for (int i = 0; i < 2; ++i) {
    ibook.setCurrentFolder(topfoldername_ + "/CaloRecHits" + caloLabels[i]);

    const std::string& lbl = caloLabels[i];
    const std::string& sfx = caloSuffixes[i];

    // rechit multiplicities are large: use 100 bins over the configurable range instead of one bin per integer
    ebRecHitsNumber_hist[i] = ibook.book1D("ebRechitsN" + sfx,
                                           "Number of EB RecHits (" + lbl + "); Number of EB recHits; Entries",
                                           100,
                                           0.0,
                                           ranges_.nEBRecHits);

    ebRecHits_energy_hist[i] =
        ibook.book1DD("ebRechits_energy" + sfx,
                      "Energy spectrum of EB RecHits (" + lbl + "); Energy of EB recHits (GeV); Entries",
                      100,
                      0.0,
                      500.0);

    ebRecHits_time_hist[i] = ibook.book1DD(
        "ebRechits_time" + sfx, "Time of EB RecHits (" + lbl + "); Time of EB recHits (ns); Entries", 200, -100., 100.0);
    eeRecHitsNumber_hist[i] = ibook.book1D("eeRechitsN" + sfx,
                                           "Number of EE RecHits (" + lbl + "); Number of EE recHits; Entries",
                                           100,
                                           0.0,
                                           ranges_.nEERecHits);
    eeRecHits_energy_hist[i] =
        ibook.book1DD("eeRechits_energy" + sfx,
                      "Energy spectrum of EE RecHits (" + lbl + "); Energy of EE recHits (GeV); Entries",
                      100,
                      0.0,
                      1000.0);
    eeRecHits_time_hist[i] = ibook.book1DD("eeRechits_time" + sfx,
                                           "Time of EE RecHits (" + lbl + "); Time of EE recHits (ns); Entries",
                                           200,
                                           -100.0,
                                           100.0);

    // EB: ieta in [-85, 85] (0 excluded), iphi in [1, 360]
    ebRecHitsEtaPhiMap[i] = ibook.book2D("ebRecHitsEtaPhiMap" + sfx,
                                         "Occupancy map of EB rechits (" + lbl + ");ieta;iphi;Entries",
                                         171,
                                         -85.5,
                                         85.5,
                                         360,
                                         0.5,
                                         360.5);

    ebRecHitsEtaPhiMap[i]->setOption("colz");

    // EE: ix, iy in [1, 100]
    eePlusRecHitsXYMap[i] = ibook.book2D("eePlusRecHitsXYMap" + sfx,
                                         "Occupancy map of EE+ rechits (" + lbl + ");ix;iy;Entries",
                                         100,
                                         0.5,
                                         100.5,
                                         100,
                                         0.5,
                                         100.5);

    eePlusRecHitsXYMap[i]->setOption("colz");

    eeMinusRecHitsXYMap[i] = ibook.book2D("eeMinusRecHitsXYMap" + sfx,
                                          "Occupancy map of EE- rechits (" + lbl + ");ix;iy;Entries",
                                          100,
                                          0.5,
                                          100.5,
                                          100,
                                          0.5,
                                          100.5);

    eeMinusRecHitsXYMap[i]->setOption("colz");
  }

  ibook.setCurrentFolder(topfoldername_ + "/CaloRecHitsAll");

  // now do HCAL (the ordering HBHE, HB, HE must match the indices used in analyze)
  const std::array<std::string, 3> subdets = {{"HBHE", "HB", "HE"}};

  // helper lambda
  auto toLower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
  };

  for (int i = 0; i < 3; ++i) {
    const std::string& subdet = subdets[i];
    const std::string name = toLower(subdet);

    // rechit multiplicities are large: use 100 bins over the configurable range instead of one bin per integer
    hbheRecHitsNumber_hist[i] =
        ibook.book1D(name + "RechitsN",
                     "Number of " + subdet + " RecHits; Number of " + subdet + " recHits; Entries",
                     100,
                     0.0,
                     ranges_.nHBHERecHits);

    hbheRecHits_energy_hist[i] =
        ibook.book1DD(name + "Rechits_energy",
                      "Energy spectrum of " + subdet + " RecHits; Energy of " + subdet + " recHits (GeV); Entries",
                      100,
                      0.0,
                      200.0);

    // Energy > 5 GeV histograms
    hbheRecHits_energy_egt5_hist[i] = ibook.book1DD(
        name + "StiffRechits_energy",
        "Energy spectrum of " + subdet + " RecHits (E > 5 GeV); Energy of stiff " + subdet + " recHits (GeV); Entries",
        100,
        0.0,
        30.0);

    hbheRecHits_time_hist[i] =
        ibook.book1DD(name + "Rechits_time",
                      "Time of " + subdet + " RecHits; Time of " + subdet + " recHits (ns); Entries",
                      100,
                      0.,
                      30.0);

    // Energy > 5 GeV histograms
    hbheRecHits_time_egt5_hist[i] =
        ibook.book1DD(name + "StiffRechits_time",
                      "Time of " + subdet + " RecHits (E > 5 GeV); Time of stiff " + subdet + " recHits (ns); Entries",
                      100,
                      0.,
                      30.0);
  }

  // HB covers |ieta| <= 16, HE covers 16 <= |ieta| <= 29; iphi in [1, 72] for both
  auto bookHcalOccupancyMap = [&ibook](const std::string& name, const std::string& subdet) {
    auto* me =
        ibook.book2D(name, "Occupancy map of " + subdet + " rechits;ieta;iphi;Entries", 59, -29.5, 29.5, 72, 0.5, 72.5);
    me->setOption("colz");
    return me;
  };

  hbheRecHitsEtaPhiMap = bookHcalOccupancyMap("hbheRecHitsEtaPhiMap", "HBHE");
  hbRecHitsEtaPhiMap = bookHcalOccupancyMap("hbRecHitsEtaPhiMap", "HB");
  heRecHitsEtaPhiMap = bookHcalOccupancyMap("heRecHitsEtaPhiMap", "HE");
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

  // Each InputTag is  ("producerLabel", "instanceLabel")
  const std::string prod = "run3ScoutingElectronBestTrack";
  desc.add<edm::InputTag>("vmBestTrackIndex", edm::InputTag(prod, "Run3ScoutingElectronBestTrackIndex"));
  desc.add<edm::InputTag>("vmTrkd0", edm::InputTag(prod, "Run3ScoutingElectronTrackd0"));
  desc.add<edm::InputTag>("vmTrkdz", edm::InputTag(prod, "Run3ScoutingElectronTrackdz"));
  desc.add<edm::InputTag>("vmTrkpt", edm::InputTag(prod, "Run3ScoutingElectronTrackpt"));
  desc.add<edm::InputTag>("vmTrketa", edm::InputTag(prod, "Run3ScoutingElectronTracketa"));
  desc.add<edm::InputTag>("vmTrkphi", edm::InputTag(prod, "Run3ScoutingElectronTrackphi"));
  desc.add<edm::InputTag>("vmTrkpMode", edm::InputTag(prod, "Run3ScoutingElectronTrackpMode"));
  desc.add<edm::InputTag>("vmTrketaMode", edm::InputTag(prod, "Run3ScoutingElectronTracketaMode"));
  desc.add<edm::InputTag>("vmTrkphiMode", edm::InputTag(prod, "Run3ScoutingElectronTrackphiMode"));
  desc.add<edm::InputTag>("vmTrkqoverpModeError", edm::InputTag(prod, "Run3ScoutingElectronTrackqoverpModeError"));
  desc.add<edm::InputTag>("vmTrkchi2overndf", edm::InputTag(prod, "Run3ScoutingElectronTrackchi2overndf"));
  desc.add<edm::InputTag>("vmTrkcharge", edm::InputTag(prod, "Run3ScoutingElectronTrackcharge"));

  desc.add<std::string>("topfoldername", "HLT/ScoutingOffline/Miscellaneous");

  // Upper edges of the multiplicity histograms (defaults tuned for Run 3).
  // Object multiplicities get one bin per integer in [0, N]; the rechit multiplicities and the
  // pile-up axis of the profiles use 100 bins and one bin per unit of pile-up, respectively.
  edm::ParameterSetDescription rangesDesc;
  rangesDesc.add<int>("nTracks", 400)->setComment("maximum number of tracks");
  rangesDesc.add<int>("nPrimaryVertices", 50)->setComment("maximum number of primary vertices");
  rangesDesc.add<int>("nDisplacedVertices", 10)->setComment("maximum number of displaced vertices (Vtx and NoVtx)");
  rangesDesc.add<int>("nMuons", 10)->setComment("maximum number of muons (Vtx and NoVtx)");
  rangesDesc.add<int>("nElectrons", 10)->setComment("maximum number of electrons");
  rangesDesc.add<int>("nPhotons", 25)->setComment("maximum number of photons");
  rangesDesc.add<int>("nPFJets", 100)->setComment("maximum number of PF jets");
  rangesDesc.add<int>("nPFCands", 1000)->setComment("maximum number of PF candidates");
  rangesDesc.add<int>("nEBRecHits", 1000)->setComment("maximum number of EB rechits");
  rangesDesc.add<int>("nEERecHits", 1000)->setComment("maximum number of EE rechits");
  rangesDesc.add<int>("nHBHERecHits", 2000)->setComment("maximum number of HBHE rechits");
  rangesDesc.add<double>("pileUp", 70.)->setComment("maximum average pile-up (x-axis of the vs-PU profiles)");
  desc.add<edm::ParameterSetDescription>("multiplicityRanges", rangesDesc);

  // Binning of the rho histogram; the [min, max] range is also used for the rho axis of the rho vs PU profile
  edm::ParameterSetDescription rhoBinningDesc;
  rhoBinningDesc.add<int>("nbins", 100)->setComment("number of bins");
  rhoBinningDesc.add<double>("min", 0.)->setComment("lower edge (GeV)");
  rhoBinningDesc.add<double>("max", 60.)->setComment("upper edge (GeV)");
  desc.add<edm::ParameterSetDescription>("rhoBinning", rhoBinningDesc);

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ScoutingCollectionMonitor);
