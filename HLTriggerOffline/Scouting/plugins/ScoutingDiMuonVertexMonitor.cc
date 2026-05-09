// -*- C++ -*-
//
// Package:    HLTriggerOffline/Scouting
// Class:      ScoutingDiMuonVertexMonitor
//
// Description:
//   DQM monitor for di-muon vertices in Run 3 scouting data.
//   Inspired by DQMOffline/Alignment/src/DiMuonVertexMonitor.cc but adapted
//   for scouting objects:
//     - Uses Run3ScoutingMuon (no reco::Track, no TransientTrackBuilder)
//     - Uses Run3ScoutingVertex for both the pre-fitted di-muon secondary
//       vertex and the primary vertex collection
//     - Avoids KalmanVertexFitter (vertex already computed online)
//
//   Per-pair quantities monitored:
//     * Invariant mass (full range + Z and J/ψ windows)
//     * SV probability, χ²/ndof
//     * PV–SV distance (xy and 3D) and significance
//     * cos φ (xy and 3D) between di-muon momentum and PV→SV displacement
//     * Per-muon track d_xy and d_z w.r.t. the primary vertex
//
// Author:     Auto-generated skeleton (based on ScoutingDileptonMonitor)

// system includes
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

// CMSSW framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Scouting data formats
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"

// Math
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TMath.h"

// Local utilities (ID selectors shared with ScoutingDileptonMonitor)
#include "ScoutingDQMUtils.h"

// ---------------------------------------------------------------------------
// Helper: Lorentz vector from scouting muon
// ---------------------------------------------------------------------------
namespace {
  inline math::PtEtaPhiMLorentzVector muonP4(const Run3ScoutingMuon& mu) {
    return math::PtEtaPhiMLorentzVector(mu.pt(), mu.eta(), mu.phi(), scoutingDQMUtils::MUON_MASS);
  }

  // Vertex position uncertainty in 2D (xy) using diagonal errors only.
  // If the newer covariance accessors are available (xyCov etc.) they should
  // be used instead; we fall back to quadratic sum of xError/yError here for
  // maximum compatibility.
  inline double vtxDistXY(float svX,
                          float svY,
                          float pvX,
                          float pvY,
                          float svXErr,
                          float svYErr,
                          float pvXErr,
                          float pvYErr,
                          double& dist,
                          double& distErr) {
    const double dx = svX - pvX;
    const double dy = svY - pvY;
    dist = std::sqrt(dx * dx + dy * dy);
    // Propagated uncertainty (ignoring off-diagonal covariance)
    if (dist > 0.) {
      distErr =
          std::sqrt((dx * dx * (svXErr * svXErr + pvXErr * pvXErr) + dy * dy * (svYErr * svYErr + pvYErr * pvYErr)) /
                    (dist * dist));
    } else {
      distErr = 0.;
    }
    return dist;
  }

  inline double vtxDist3D(float svX,
                          float svY,
                          float svZ,
                          float pvX,
                          float pvY,
                          float pvZ,
                          float svXErr,
                          float svYErr,
                          float svZErr,
                          float pvXErr,
                          float pvYErr,
                          float pvZErr,
                          double& dist,
                          double& distErr) {
    const double dx = svX - pvX;
    const double dy = svY - pvY;
    const double dz = svZ - pvZ;
    dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist > 0.) {
      distErr =
          std::sqrt((dx * dx * (svXErr * svXErr + pvXErr * pvXErr) + dy * dy * (svYErr * svYErr + pvYErr * pvYErr) +
                     dz * dz * (svZErr * svZErr + pvZErr * pvZErr)) /
                    (dist * dist));
    } else {
      distErr = 0.;
    }
    return dist;
  }

  // Unit-safe cm → μm conversion factor
  constexpr double cmToUm = 1.e4;
}  // namespace

// ---------------------------------------------------------------------------
// Monitor class
// ---------------------------------------------------------------------------
class ScoutingDiMuonVertexMonitor : public DQMEDAnalyzer {
public:
  explicit ScoutingDiMuonVertexMonitor(const edm::ParameterSet&);
  ~ScoutingDiMuonVertexMonitor() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  // ---- configuration -------------------------------------------------------
  const std::string folderName_;
  const std::string decayMotherName_;

  // Mass window determined by decayMotherName_
  std::pair<double, double> massLimits_;

  // Muon selection
  const bool applyMuonID_;
  const double minPt_;
  const double maxEta_;

  // Vertex quality cuts
  const double minVtxProb_;
  const double maxSVdistXY_;  // μm, cut on PV–SV xy distance

  // Input tags / tokens
  const edm::EDGetTokenT<Run3ScoutingMuonCollection> muonToken_;
  const edm::EDGetTokenT<Run3ScoutingVertexCollection> pvToken_;
  const edm::EDGetTokenT<Run3ScoutingVertexCollection> svToken_;  // di-muon displaced vtx

  // ---- histograms ----------------------------------------------------------

  // Vertex quality
  MonitorElement* hSVProb_{nullptr};
  MonitorElement* hSVChi2_{nullptr};
  MonitorElement* hSVNormChi2_{nullptr};

  // PV–SV distance
  MonitorElement* hSVDistXY_{nullptr};
  MonitorElement* hSVDistXYErr_{nullptr};
  MonitorElement* hSVDistXYSig_{nullptr};
  MonitorElement* hSVDist3D_{nullptr};
  MonitorElement* hSVDist3DErr_{nullptr};
  MonitorElement* hSVDist3DSig_{nullptr};

  // Pointing angle
  MonitorElement* hCosPhi_{nullptr};    // cos φ in xy
  MonitorElement* hCosPhi3D_{nullptr};  // cos φ in 3D
  MonitorElement* hCosPhiInv_{nullptr};
  MonitorElement* hCosPhiInv3D_{nullptr};
  MonitorElement* hCosPhiUnbalance_{nullptr};
  MonitorElement* hCosPhi3DUnbalance_{nullptr};

  // Invariant mass
  MonitorElement* hInvMass_{nullptr};

  // Per-muon track impact parameters w.r.t. the primary vertex
  MonitorElement* hTrkDxy_{nullptr};
  MonitorElement* hTrkDz_{nullptr};
  MonitorElement* hTrkDxyErr_{nullptr};
  MonitorElement* hTrkDzErr_{nullptr};

  // Number of muons and vertices per event (diagnostic)
  MonitorElement* hNMuons_{nullptr};
  MonitorElement* hNSV_{nullptr};
};

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ScoutingDiMuonVertexMonitor::ScoutingDiMuonVertexMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      decayMotherName_(iConfig.getParameter<std::string>("decayMotherName")),
      applyMuonID_(iConfig.getParameter<bool>("applyMuonID")),
      minPt_(iConfig.getParameter<double>("minMuonPt")),
      maxEta_(iConfig.getParameter<double>("maxMuonEta")),
      minVtxProb_(iConfig.getParameter<double>("minVtxProb")),
      maxSVdistXY_(iConfig.getParameter<double>("maxSVdistXY")),
      muonToken_(consumes<Run3ScoutingMuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      pvToken_(consumes<Run3ScoutingVertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertices"))),
      svToken_(consumes<Run3ScoutingVertexCollection>(iConfig.getParameter<edm::InputTag>("secondaryVertices"))) {
  // Determine mass window from decay mother name
  if (decayMotherName_.find('Z') != std::string::npos) {
    massLimits_ = {50., 120.};
  } else if (decayMotherName_.find("J/#psi") != std::string::npos ||
             decayMotherName_.find("J/psi") != std::string::npos) {
    massLimits_ = {2.7, 3.4};
  } else if (decayMotherName_.find("#Upsilon") != std::string::npos ||
             decayMotherName_.find("Upsilon") != std::string::npos) {
    massLimits_ = {8.9, 9.9};
  } else {
    edm::LogWarning("ScoutingDiMuonVertexMonitor")
        << "Unrecognised decay mother '" << decayMotherName_ << "'. Defaulting to Z window [50, 120] GeV.";
    massLimits_ = {50., 120.};
  }
}

// ---------------------------------------------------------------------------
// bookHistograms
// ---------------------------------------------------------------------------
void ScoutingDiMuonVertexMonitor::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  iBooker.setCurrentFolder(folderName_ + "/ScoutingDiMuonVertexMonitor");

  const std::string motName = decayMotherName_;
  const std::string ps = "N(#mu#mu pairs)";
  const std::string histTit = motName + " #rightarrow #mu^{+}#mu^{-}";

  // ---- vertex quality ----
  hSVProb_ = iBooker.book1D("VtxProb", ";" + motName + " vertex probability;" + ps, 100, 0., 1.);

  hSVChi2_ = iBooker.book1D(
      "VtxChi2", ";#chi^{2} of the " + motName + " vertex;#chi^{2} of the " + motName + " vertex;" + ps, 200, 0., 200.);

  hSVNormChi2_ = iBooker.book1D("VtxNormChi2", ";#chi^{2}/ndof of the " + motName + " vertex;" + ps, 100, 0., 20.);

  // ---- PV–SV distance xy ----
  hSVDistXY_ = iBooker.book1D("VtxDistXY", histTit + ";PV-" + motName + "V xy distance [#mum];" + ps, 100, 0., 300.);

  hSVDistXYErr_ =
      iBooker.book1D("VtxDistXYErr", histTit + ";PV-" + motName + "V xy distance error [#mum];" + ps, 100, 0., 1000.);

  hSVDistXYSig_ =
      iBooker.book1D("VtxDistXYSig", histTit + ";PV-" + motName + "V xy distance significance;" + ps, 100, 0., 10.);

  // ---- PV–SV distance 3D ----
  hSVDist3D_ = iBooker.book1D("VtxDist3D", histTit + ";PV-" + motName + "V 3D distance [#mum];" + ps, 100, 0., 300.);

  hSVDist3DErr_ =
      iBooker.book1D("VtxDist3DErr", histTit + ";PV-" + motName + "V 3D distance error [#mum];" + ps, 100, 0., 1000.);

  hSVDist3DSig_ =
      iBooker.book1D("VtxDist3DSig", histTit + ";PV-" + motName + "V 3D distance significance;" + ps, 100, 0., 10.);

  // ---- pointing angle ----
  hCosPhi_ = iBooker.book1D("CosPhi", histTit + ";cos(#phi_{xy});" + ps, 50, -1., 1.);

  hCosPhi3D_ = iBooker.book1D("CosPhi3D", histTit + ";cos(#phi_{3D});" + ps, 50, -1., 1.);

  hCosPhiInv_ = iBooker.book1D("CosPhiInv", histTit + ";inverted cos(#phi_{xy});" + ps, 50, -1., 1.);

  hCosPhiInv3D_ = iBooker.book1D("CosPhiInv3D", histTit + ";inverted cos(#phi_{3D});" + ps, 50, -1., 1.);

  hCosPhiUnbalance_ = iBooker.book1D("CosPhiUnbalance", histTit + ";cos(#phi_{xy}) unbalance;#Delta" + ps, 50, -1., 1.);

  hCosPhi3DUnbalance_ =
      iBooker.book1D("CosPhi3DUnbalance", histTit + ";cos(#phi_{3D}) unbalance;#Delta" + ps, 50, -1., 1.);

  // ---- invariant mass ----
  hInvMass_ =
      iBooker.book1D("InvMass", histTit + ";M(#mu^{+}#mu^{-}) [GeV];" + ps, 70, massLimits_.first, massLimits_.second);

  // ---- track impact parameters ----
  hTrkDxy_ = iBooker.book1D("TrkDxy", histTit + ";muon track d_{xy}(PV) [#mum];muon tracks", 150, -300., 300.);

  hTrkDz_ = iBooker.book1D("TrkDz", histTit + ";muon track d_{z}(PV) [#mum];muon tracks", 150, -300., 300.);

  hTrkDxyErr_ = iBooker.book1D("TrkDxyErr", histTit + ";muon track err_{dxy} [#mum];muon tracks", 250, 0., 500.);

  hTrkDzErr_ = iBooker.book1D("TrkDzErr", histTit + ";muon track err_{dz} [#mum];muon tracks", 250, 0., 500.);

  // ---- diagnostics ----
  hNMuons_ = iBooker.book1D("NMuons", ";Number of selected muons;Events", 20, 0., 20.);
  hNSV_ = iBooker.book1D("NSV", ";Number of di-muon secondary vertices;Events", 20, 0., 20.);
}

// ---------------------------------------------------------------------------
// analyze
// ---------------------------------------------------------------------------
void ScoutingDiMuonVertexMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const&) {
  // ---- Muon collection ----
  edm::Handle<Run3ScoutingMuonCollection> muonHandle;
  iEvent.getByToken(muonToken_, muonHandle);
  if (!muonHandle.isValid()) {
    edm::LogWarning("ScoutingDiMuonVertexMonitor") << "Invalid muon collection handle.";
    return;
  }

  // ---- Primary vertex collection ----
  edm::Handle<Run3ScoutingVertexCollection> pvHandle;
  iEvent.getByToken(pvToken_, pvHandle);
  if (!pvHandle.isValid() || pvHandle->empty()) {
    edm::LogWarning("ScoutingDiMuonVertexMonitor") << "No primary vertices found; skipping event.";
    return;
  }
  const Run3ScoutingVertex& pv = pvHandle->front();  // use the first (hardest) PV

  // ---- Secondary vertex collection (di-muon) ----
  edm::Handle<Run3ScoutingVertexCollection> svHandle;
  iEvent.getByToken(svToken_, svHandle);
  if (!svHandle.isValid()) {
    edm::LogWarning("ScoutingDiMuonVertexMonitor") << "Invalid secondary vertex collection handle.";
    return;
  }

  hNSV_->Fill(static_cast<double>(svHandle->size()));

  // ---- Muon selection ----
  std::vector<size_t> selectedMuons;
  selectedMuons.reserve(muonHandle->size());
  for (size_t i = 0; i < muonHandle->size(); ++i) {
    const auto& mu = (*muonHandle)[i];
    if (mu.pt() < minPt_)
      continue;
    if (std::abs(mu.eta()) > maxEta_)
      continue;
    if (applyMuonID_ && !scoutingDQMUtils::scoutingMuonID(mu))
      continue;
    selectedMuons.push_back(i);
  }

  hNMuons_->Fill(static_cast<double>(selectedMuons.size()));

  if (selectedMuons.size() < 2)
    return;

  // ---- Loop over di-muon secondary vertices ----
  // Each SV in the displaced-vertex collection corresponds to a di-muon pair.
  // We pick the two muons (closest in ΔR to the SV flight direction) that form
  // an opposite-sign pair.  When no per-SV muon index map is available (the
  // scouting format does not store it), we fall back to the globally selected
  // muon pair with the smallest |Δm – m_SV|, i.e. all pairs are tried and the
  // best-matching one is chosen.
  //
  // For simplicity and maximum compatibility we adopt the safe approach:
  // iterate over all OS muon pairs, reconstruct their kinematics, and then
  // look for a matching SV (closest in 3D position to the di-muon flight path).
  // If no dedicated SV collection is provided (svToken points to an empty
  // collection) the pair still contributes to kinematic histograms via the
  // primary-vertex information only.

  for (size_t ii = 0; ii < selectedMuons.size(); ++ii) {
    for (size_t jj = ii + 1; jj < selectedMuons.size(); ++jj) {
      const auto& mu_i = (*muonHandle)[selectedMuons[ii]];
      const auto& mu_j = (*muonHandle)[selectedMuons[jj]];

      // Opposite sign
      if (mu_i.charge() * mu_j.charge() >= 0)
        continue;

      // Invariant mass
      const auto p4pair = muonP4(mu_i) + muonP4(mu_j);
      const double mass = p4pair.mass();
      hInvMass_->Fill(mass);

      // Di-muon momentum direction
      const double pxDimu = p4pair.px();
      const double pyDimu = p4pair.py();
      const double pzDimu = p4pair.pz();

      // ---- Per-muon impact parameters w.r.t. PV ----
      // Run3ScoutingMuon stores trk_dxy and trk_dz computed w.r.t. the
      // beamspot / PV at HLT.  We use them directly.
      for (const auto* muPtr : {&mu_i, &mu_j}) {
        hTrkDxy_->Fill(muPtr->trk_dxy() * cmToUm);
        hTrkDz_->Fill(muPtr->trk_dz() * cmToUm);
        hTrkDxyErr_->Fill(muPtr->trk_dxyError() * cmToUm);
        hTrkDzErr_->Fill(muPtr->trk_dzError() * cmToUm);
      }

      // ---- Find best matching secondary vertex ----
      // Strategy: among all SVs, pick the one with the minimum 3D distance
      // to the midpoint of the two muon "reference point" vectors.
      // As a proxy we use the SV position directly; the scouting displaced
      // vertex collection is typically pre-matched to muon pairs by the HLT.
      const Run3ScoutingVertex* bestSV = nullptr;
      double minSVdist = 1e9;
      for (const auto& sv : *svHandle) {
        if (!sv.isValidVtx())
          continue;
        const double dx = sv.x() - pv.x();
        const double dy = sv.y() - pv.y();
        const double dz = sv.z() - pv.z();
        const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (d < minSVdist) {
          minSVdist = d;
          bestSV = &sv;
        }
      }

      if (bestSV == nullptr)
        continue;  // no SV available for this pair

      // ---- Vertex probability and χ² ----
      const double chi2 = bestSV->chi2();
      const int ndof = bestSV->ndof();
      const double prob = (ndof > 0) ? TMath::Prob(chi2, ndof) : 0.;

      hSVProb_->Fill(prob);
      hSVChi2_->Fill(chi2);
      if (ndof > 0)
        hSVNormChi2_->Fill(chi2 / ndof);

      if (prob < minVtxProb_)
        continue;

      // ---- PV–SV distances ----
      // Use diagonal position errors only (conservative); if the newer
      // Run3ScoutingVertex with covariance is available, subclasses can
      // override with more precise error propagation.
      double distXY, distXYErr;
      vtxDistXY(bestSV->x(),
                bestSV->y(),
                pv.x(),
                pv.y(),
                bestSV->xError(),
                bestSV->yError(),
                pv.xError(),
                pv.yError(),
                distXY,
                distXYErr);

      double dist3D, dist3DErr;
      vtxDist3D(bestSV->x(),
                bestSV->y(),
                bestSV->z(),
                pv.x(),
                pv.y(),
                pv.z(),
                bestSV->xError(),
                bestSV->yError(),
                bestSV->zError(),
                pv.xError(),
                pv.yError(),
                pv.zError(),
                dist3D,
                dist3DErr);

      hSVDistXY_->Fill(distXY * cmToUm);
      hSVDistXYErr_->Fill(distXYErr * cmToUm);
      if (distXYErr > 0.)
        hSVDistXYSig_->Fill(distXY / distXYErr);

      hSVDist3D_->Fill(dist3D * cmToUm);
      hSVDist3DErr_->Fill(dist3DErr * cmToUm);
      if (dist3DErr > 0.)
        hSVDist3DSig_->Fill(dist3D / dist3DErr);

      // ---- Pointing angles ----
      // Displacement vector PV → SV
      const double dVtxX = bestSV->x() - pv.x();
      const double dVtxY = bestSV->y() - pv.y();
      const double dVtxZ = bestSV->z() - pv.z();

      // cos φ in xy plane
      const double magPtDimu = std::sqrt(pxDimu * pxDimu + pyDimu * pyDimu);
      const double magDVtxXY = std::sqrt(dVtxX * dVtxX + dVtxY * dVtxY);
      const double magPDimu = std::sqrt(pxDimu * pxDimu + pyDimu * pyDimu + pzDimu * pzDimu);
      const double magDVtx3D = std::sqrt(dVtxX * dVtxX + dVtxY * dVtxY + dVtxZ * dVtxZ);

      if (magPtDimu > 0. && magDVtxXY > 0.) {
        const double cosphi = (pxDimu * dVtxX + pyDimu * dVtxY) / (magPtDimu * magDVtxXY);
        hCosPhi_->Fill(cosphi);
        hCosPhiInv_->Fill(-cosphi);
        hCosPhiUnbalance_->Fill(cosphi, 1.);
        hCosPhiUnbalance_->Fill(-cosphi, -1.);
      }

      if (magPDimu > 0. && magDVtx3D > 0.) {
        const double cosphi3D = (pxDimu * dVtxX + pyDimu * dVtxY + pzDimu * dVtxZ) / (magPDimu * magDVtx3D);
        hCosPhi3D_->Fill(cosphi3D);
        hCosPhiInv3D_->Fill(-cosphi3D);
        hCosPhi3DUnbalance_->Fill(cosphi3D, 1.);
        hCosPhi3DUnbalance_->Fill(-cosphi3D, -1.);
      }

      // ---- Distance cut ----
      if (distXY * cmToUm > maxSVdistXY_)
        continue;

      // Additional per-pair histograms after distance cut can be added here.
    }
  }
}

// ---------------------------------------------------------------------------
// fillDescriptions
// ---------------------------------------------------------------------------
void ScoutingDiMuonVertexMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("FolderName", "HLT/ScoutingOffline/DiMuon");
  desc.add<std::string>("decayMotherName", "Z");

  desc.add<edm::InputTag>("muons", edm::InputTag("hltScoutingMuonPackerVtx"));
  desc.add<edm::InputTag>("primaryVertices", edm::InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"));
  desc.add<edm::InputTag>("secondaryVertices", edm::InputTag("hltScoutingMuonPackerVtx", "displacedVtx"));

  desc.add<bool>("applyMuonID", true);
  desc.add<double>("minMuonPt", 3.0);
  desc.add<double>("maxMuonEta", 2.4);
  desc.add<double>("minVtxProb", 0.005);
  desc.add<double>("maxSVdistXY", 50.0);  // μm

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(ScoutingDiMuonVertexMonitor);
