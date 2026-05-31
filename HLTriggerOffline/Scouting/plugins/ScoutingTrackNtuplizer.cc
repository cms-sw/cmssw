// -*- C++ -*-
//
// Package:    HLTriggerOffline/Scouting
// Class:      ScoutingTrackNtuplizer
//
// Description: EDAnalyzer that dumps a TTree with track/vertex quantities
//
// Usage: add to your cmsRun config, open output ROOT file, inspect the "trackTree" TTree.

// system
#include <cmath>
#include <memory>
#include <string>
#include <vector>

// CMSSW framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

// data formats
#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Vector3D.h"

// ROOT / TFileService
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"
#include "TMath.h"

class ScoutingTrackNtuplizer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ScoutingTrackNtuplizer(const edm::ParameterSet&);
  ~ScoutingTrackNtuplizer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  // helpers (same as ScoutingTrackMonitor)
  reco::Track makeRecoTrack(const Run3ScoutingTrack&) const;
  reco::Vertex makeRecoVertex(const Run3ScoutingVertex&) const;
  std::pair<unsigned int, const Run3ScoutingVertex*> findClosestVertex(const reco::Track&,
                                                                       const std::vector<Run3ScoutingVertex>&) const;

  // tokens
  const edm::EDGetTokenT<std::vector<Run3ScoutingTrack>> tracksToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> verticesToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

  static constexpr float cmToUm = 10000.f;

  // TTree and branches
  TTree* tree_ = nullptr;

  // --- per-track branches ---

  // kinematics
  float b_pt, b_eta, b_phi;

  // stored dz/dxy (directly from scouting track, w.r.t. beam line)
  float b_dz_stored;      // trk.tk_dz()       [µm]
  float b_dxy_stored;     // trk.tk_dxy()      [µm]
  float b_dzErr_stored;   // trk.tk_dz_Error() [µm]
  float b_dxyErr_stored;  // trk.tk_dxy_Error()[µm]

  // recomputed dz/dxy w.r.t. closest vertex
  float b_dz_vtx;   // recoTrk.dz(vtx)   [µm]
  float b_dxy_vtx;  // recoTrk.dxy(vtx)  [µm]

  // track uncertainties from covariance matrix
  float b_dzErr_track;   // recoTrk.dzError() [µm]
  float b_dxyErr_track;  // recoTrk.dxyError()[µm]

  // vertex uncertainties
  float b_vtx_zErr;  // closestVtx.zError()  [µm]
  float b_vtx_xErr;  // closestVtx.xError()  [µm]
  float b_vtx_yErr;  // closestVtx.yError()  [µm]

  // combined uncertainties (track + vtx in quadrature)
  float b_dzErr_combined;
  float b_dxyErr_combined;

  // pulls
  float b_dzPull_stored;    // dz_stored   / dzErr_stored
  float b_dzPull_vtx;       // dz_vtx      / dzErr_track
  float b_dzPull_combined;  // dz_vtx      / dzErr_combined
  float b_dxyPull_stored;
  float b_dxyPull_vtx;
  float b_dxyPull_combined;

  // track reference point (stored vx/vy/vz from packer)
  float b_vx_stored;  // trk.tk_vx() [cm]
  float b_vy_stored;  // trk.tk_vy() [cm]
  float b_vz_stored;  // trk.tk_vz() [cm]
  float b_vr_stored;  // sqrt(vx^2+vy^2) [cm] — production radius

  // closest vertex position
  float b_vtx_x, b_vtx_y, b_vtx_z;
  int b_vtx_index;
  int b_vtx_ntracks;

  // delta between stored reference point and vertex
  float b_delta_vz;  // vz_stored - vtx_z  [cm]
  float b_delta_vr;  // vr_stored           [cm] (distance from beam line)

  // track quality
  float b_chi2;
  float b_normchi2;
  int b_ndof;
  int b_nPixelHits;
  int b_nTrackerLayers;
  int b_nStripHits;
  int b_charge;

  // beamspot
  float b_bs_x, b_bs_y, b_bs_z;

  // event info
  unsigned int b_run, b_lumi;
  unsigned long long b_event;

  // flag: is this track in the spike (|dzPull_combined| < 0.3)?
  int b_is_spike;
};

// ---------------------------------------------------------------------------
ScoutingTrackNtuplizer::ScoutingTrackNtuplizer(const edm::ParameterSet& iConfig)
    : tracksToken_{consumes<std::vector<Run3ScoutingTrack>>(iConfig.getParameter<edm::InputTag>("tracks"))},
      verticesToken_{consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("vertices"))},
      beamSpotToken_{consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotLabel"))} {
  usesResource("TFileService");
}

// ---------------------------------------------------------------------------
void ScoutingTrackNtuplizer::beginJob() {
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("trackTree", "Scouting track debug tree");

  // event
  tree_->Branch("run", &b_run);
  tree_->Branch("lumi", &b_lumi);
  tree_->Branch("event", &b_event);

  // kinematics
  tree_->Branch("pt", &b_pt);
  tree_->Branch("eta", &b_eta);
  tree_->Branch("phi", &b_phi);

  // stored (packer) quantities
  tree_->Branch("dz_stored", &b_dz_stored);
  tree_->Branch("dxy_stored", &b_dxy_stored);
  tree_->Branch("dzErr_stored", &b_dzErr_stored);
  tree_->Branch("dxyErr_stored", &b_dxyErr_stored);

  // recomputed w.r.t. vertex
  tree_->Branch("dz_vtx", &b_dz_vtx);
  tree_->Branch("dxy_vtx", &b_dxy_vtx);

  // track-only uncertainties
  tree_->Branch("dzErr_track", &b_dzErr_track);
  tree_->Branch("dxyErr_track", &b_dxyErr_track);

  // vertex uncertainties
  tree_->Branch("vtx_zErr", &b_vtx_zErr);
  tree_->Branch("vtx_xErr", &b_vtx_xErr);
  tree_->Branch("vtx_yErr", &b_vtx_yErr);

  // combined uncertainties
  tree_->Branch("dzErr_combined", &b_dzErr_combined);
  tree_->Branch("dxyErr_combined", &b_dxyErr_combined);

  // pulls (three variants so you can compare)
  tree_->Branch("dzPull_stored", &b_dzPull_stored);
  tree_->Branch("dzPull_vtx", &b_dzPull_vtx);
  tree_->Branch("dzPull_combined", &b_dzPull_combined);
  tree_->Branch("dxyPull_stored", &b_dxyPull_stored);
  tree_->Branch("dxyPull_vtx", &b_dxyPull_vtx);
  tree_->Branch("dxyPull_combined", &b_dxyPull_combined);

  // stored reference point
  tree_->Branch("vx_stored", &b_vx_stored);
  tree_->Branch("vy_stored", &b_vy_stored);
  tree_->Branch("vz_stored", &b_vz_stored);
  tree_->Branch("vr_stored", &b_vr_stored);  // transverse production radius

  // closest vertex
  tree_->Branch("vtx_x", &b_vtx_x);
  tree_->Branch("vtx_y", &b_vtx_y);
  tree_->Branch("vtx_z", &b_vtx_z);
  tree_->Branch("vtx_index", &b_vtx_index);
  tree_->Branch("vtx_ntracks", &b_vtx_ntracks);

  // deltas between stored ref point and vertex
  tree_->Branch("delta_vz", &b_delta_vz);  // vz_stored - vtx_z: should be ~0 for prompt
  tree_->Branch("delta_vr", &b_delta_vr);  // vr_stored: non-zero means non-prompt ref point

  // track quality
  tree_->Branch("chi2", &b_chi2);
  tree_->Branch("normchi2", &b_normchi2);
  tree_->Branch("ndof", &b_ndof);
  tree_->Branch("nPixelHits", &b_nPixelHits);
  tree_->Branch("nTrackerLayers", &b_nTrackerLayers);
  tree_->Branch("nStripHits", &b_nStripHits);
  tree_->Branch("charge", &b_charge);

  // beamspot
  tree_->Branch("bs_x", &b_bs_x);
  tree_->Branch("bs_y", &b_bs_y);
  tree_->Branch("bs_z", &b_bs_z);

  // spike flag
  tree_->Branch("is_spike", &b_is_spike);
}

// ---------------------------------------------------------------------------
void ScoutingTrackNtuplizer::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<std::vector<Run3ScoutingTrack>> tracksH;
  edm::Handle<std::vector<Run3ScoutingVertex>> verticesH;
  edm::Handle<reco::BeamSpot> bsH;

  iEvent.getByToken(tracksToken_, tracksH);
  iEvent.getByToken(verticesToken_, verticesH);
  iEvent.getByToken(beamSpotToken_, bsH);

  if (!tracksH.isValid() || !verticesH.isValid() || !bsH.isValid())
    return;

  const auto& vertices = *verticesH;
  if (vertices.empty())
    return;

  b_run = iEvent.id().run();
  b_lumi = iEvent.id().luminosityBlock();
  b_event = iEvent.id().event();

  b_bs_x = bsH->x0();
  b_bs_y = bsH->y0();
  b_bs_z = bsH->z0();

  for (const auto& trk : *tracksH) {
    reco::Track recoTrk = makeRecoTrack(trk);
    auto [vtxIdx, closestVtx] = findClosestVertex(recoTrk, vertices);
    if (!closestVtx)
      continue;

    reco::Vertex recoVtx = makeRecoVertex(*closestVtx);

    // --- kinematics ---
    b_pt = trk.tk_pt();
    b_eta = trk.tk_eta();
    b_phi = trk.tk_phi();

    // --- stored quantities (packer, w.r.t. beam line ref point) ---
    b_dz_stored = trk.tk_dz() * cmToUm;
    b_dxy_stored = trk.tk_dxy() * cmToUm;
    b_dzErr_stored = trk.tk_dz_Error() * cmToUm;
    b_dxyErr_stored = trk.tk_dxy_Error() * cmToUm;

    // --- recomputed w.r.t. closest vertex ---
    b_dz_vtx = recoTrk.dz(recoVtx.position()) * cmToUm;
    b_dxy_vtx = recoTrk.dxy(recoVtx.position()) * cmToUm;

    // --- uncertainties ---
    b_dzErr_track = recoTrk.dzError() * cmToUm;
    b_dxyErr_track = recoTrk.dxyError() * cmToUm;

    b_vtx_zErr = closestVtx->zError() * cmToUm;
    b_vtx_xErr = closestVtx->xError() * cmToUm;
    b_vtx_yErr = closestVtx->yError() * cmToUm;

    b_dzErr_combined = std::sqrt(b_dzErr_track * b_dzErr_track + b_vtx_zErr * b_vtx_zErr);
    b_dxyErr_combined = std::sqrt(b_dxyErr_track * b_dxyErr_track + b_vtx_xErr * b_vtx_xErr + b_vtx_yErr * b_vtx_yErr);

    // --- pulls (all three variants — compare these to find the right one) ---
    auto safeDivide = [](float num, float den) -> float { return (std::abs(den) > 1e-6f) ? num / den : -999.f; };

    b_dzPull_stored = safeDivide(b_dz_stored, b_dzErr_stored);
    b_dzPull_vtx = safeDivide(b_dz_vtx, b_dzErr_track);
    b_dzPull_combined = safeDivide(b_dz_vtx, b_dzErr_combined);

    b_dxyPull_stored = safeDivide(b_dxy_stored, b_dxyErr_stored);
    b_dxyPull_vtx = safeDivide(b_dxy_vtx, b_dxyErr_track);
    b_dxyPull_combined = safeDivide(b_dxy_vtx, b_dxyErr_combined);

    // --- stored reference point ---
    b_vx_stored = trk.tk_vx();
    b_vy_stored = trk.tk_vy();
    b_vz_stored = trk.tk_vz();
    b_vr_stored = std::sqrt(trk.tk_vx() * trk.tk_vx() + trk.tk_vy() * trk.tk_vy());

    // --- closest vertex ---
    b_vtx_x = closestVtx->x();
    b_vtx_y = closestVtx->y();
    b_vtx_z = closestVtx->z();
    b_vtx_index = static_cast<int>(vtxIdx);
    b_vtx_ntracks = closestVtx->tracksSize();

    // --- deltas: key diagnostic quantities ---
    // delta_vz != 0 means stored ref point is displaced from vertex in z
    // delta_vr != 0 means track has non-prompt production radius
    b_delta_vz = trk.tk_vz() - closestVtx->z();
    b_delta_vr = b_vr_stored;

    // --- quality ---
    b_chi2 = recoTrk.chi2();
    b_normchi2 = recoTrk.normalizedChi2();
    b_ndof = recoTrk.ndof();
    b_nPixelHits = trk.tk_nValidPixelHits();
    b_nTrackerLayers = trk.tk_nTrackerLayersWithMeasurement();
    b_nStripHits = trk.tk_nValidStripHits();
    b_charge = trk.tk_charge();

    // spike flag: use combined pull
    b_is_spike = (std::abs(b_dzPull_combined) < 0.3f) ? 1 : 0;

    tree_->Fill();
  }
}

// ---------------------------------------------------------------------------
reco::Track ScoutingTrackNtuplizer::makeRecoTrack(const Run3ScoutingTrack& sTrack) const {
  reco::Track::Point v(sTrack.tk_vx(), sTrack.tk_vy(), sTrack.tk_vz());
  reco::Track::Vector p(math::RhoEtaPhiVector(sTrack.tk_pt(), sTrack.tk_eta(), sTrack.tk_phi()));

  reco::TrackBase::CovarianceMatrix cov;
  cov(0, 0) = std::pow(sTrack.tk_qoverp_Error(), 2);
  cov(0, 1) = sTrack.tk_qoverp_lambda_cov();
  cov(0, 2) = sTrack.tk_qoverp_phi_cov();
  cov(0, 3) = sTrack.tk_qoverp_dxy_cov();
  cov(0, 4) = sTrack.tk_qoverp_dsz_cov();
  cov(1, 1) = std::pow(sTrack.tk_lambda_Error(), 2);
  cov(1, 2) = sTrack.tk_lambda_phi_cov();
  cov(1, 3) = sTrack.tk_lambda_dxy_cov();
  cov(1, 4) = sTrack.tk_lambda_dsz_cov();
  cov(2, 2) = std::pow(sTrack.tk_phi_Error(), 2);
  cov(2, 3) = sTrack.tk_phi_dxy_cov();
  cov(2, 4) = sTrack.tk_phi_dsz_cov();
  cov(3, 3) = std::pow(sTrack.tk_dxy_Error(), 2);
  cov(3, 4) = sTrack.tk_dxy_dsz_cov();
  cov(4, 4) = std::pow(sTrack.tk_dsz_Error(), 2);

  return reco::Track(sTrack.tk_chi2(), sTrack.tk_ndof(), v, p, sTrack.tk_charge(), cov);
}

// ---------------------------------------------------------------------------
reco::Vertex ScoutingTrackNtuplizer::makeRecoVertex(const Run3ScoutingVertex& sVertex) const {
  reco::Vertex::Error err;
  err(0, 0) = std::pow(sVertex.xError(), 2);
  err(1, 1) = std::pow(sVertex.yError(), 2);
  err(2, 2) = std::pow(sVertex.zError(), 2);
  err(0, 1) = sVertex.xyCov();
  err(0, 2) = sVertex.xzCov();
  err(1, 2) = sVertex.yzCov();

  return reco::Vertex(reco::Vertex::Point(sVertex.x(), sVertex.y(), sVertex.z()),
                      err,
                      sVertex.chi2(),
                      sVertex.ndof(),
                      sVertex.tracksSize());
}

// ---------------------------------------------------------------------------
std::pair<unsigned int, const Run3ScoutingVertex*> ScoutingTrackNtuplizer::findClosestVertex(
    const reco::Track& track, const std::vector<Run3ScoutingVertex>& vertices) const {
  double minDist = std::numeric_limits<double>::max();
  const Run3ScoutingVertex* closest = nullptr;
  unsigned int idx = 0, bestIdx = 999;

  for (const auto& vtx : vertices) {
    math::XYZPoint vtxPos(vtx.x(), vtx.y(), vtx.z());
    const auto& mom = track.momentum();
    const auto d = vtxPos - track.referencePoint();
    double dist = d.Cross(mom).R() / mom.R();
    if (dist < minDist) {
      minDist = dist;
      closest = &vtx;
      bestIdx = idx;
    }
    ++idx;
  }
  return {bestIdx, closest};
}

// ---------------------------------------------------------------------------
void ScoutingTrackNtuplizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracks", edm::InputTag("hltScoutingTrackPacker"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"));
  desc.add<edm::InputTag>("beamSpotLabel", edm::InputTag("hltOnlineBeamSpot"));
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ScoutingTrackNtuplizer);
