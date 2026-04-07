// system includes
#include <cmath>
#include <vector>
#include <numbers>

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ScoutingTrackMonitor : public DQMEDAnalyzer {
public:
  explicit ScoutingTrackMonitor(const edm::ParameterSet&);
  ~ScoutingTrackMonitor() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  // tokens
  const edm::EDGetTokenT<std::vector<Run3ScoutingTrack>> tracksToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> verticesToken_;

  const std::string topFolderName_;  // top folder name where to book histograms

  // histograms
  MonitorElement* h_dxy;
  MonitorElement* h_dz;
  MonitorElement* h_vtx_idx;
  MonitorElement* p_dxy_eta;
  MonitorElement* p_dxy_phi;
  MonitorElement* p_dz_eta;
  MonitorElement* p_dz_phi;

  MonitorElement* h2_eta_phi;

  // 2D eta-phi profiles
  MonitorElement* p2_dxy_eta_phi;
  MonitorElement* p2_dz_eta_phi;
  MonitorElement* p2_nValidPixelHits_eta_phi;
  MonitorElement* p2_nTrackerLayersWithMeasurement_eta_phi;
  MonitorElement* p2_nValidStripHits_eta_phi;

  static constexpr int cmToUm = 10000;

  // helpers
  reco::Track makeRecoTrack(const Run3ScoutingTrack& sTrack) const;
  reco::Vertex makeRecoVertex(const Run3ScoutingVertex& sVertex) const;
  std::pair<unsigned int, const Run3ScoutingVertex*> findClosestScoutingVertex(
      const reco::Track* track, const std::vector<Run3ScoutingVertex>& vertices);
};

// constructor
ScoutingTrackMonitor::ScoutingTrackMonitor(const edm::ParameterSet& iConfig)
    : tracksToken_{consumes<std::vector<Run3ScoutingTrack>>(iConfig.getParameter<edm::InputTag>("tracks"))},
      verticesToken_{consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("vertices"))},
      topFolderName_{iConfig.getParameter<std::string>("topFolderName")} {}

// histogram booking
void ScoutingTrackMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(topFolderName_);

  h_dxy = ibooker.book1D("dxy", "d_{xy};d_{xy} [#mum];Tracks", 100, -0.01 * cmToUm, 0.01 * cmToUm);
  h_dz = ibooker.book1D("dz", "d_{z};d_{z} [#mum];Tracks", 100, -0.05 * cmToUm, 0.05 * cmToUm);

  h_vtx_idx = ibooker.book1DD("vertexIndex", "tracks Vertex Index;Vertex index;Tracks", 17, -1.5, 15.5);

  p_dxy_eta = ibooker.bookProfile(
      "dxy_vs_eta", "d_{xy} vs #eta;#eta;#LTd_{xy}#GT [#mum]", 50, -3.0, 3.0, -0.01 * cmToUm, 0.01 * cmToUm, "");
  p_dxy_phi = ibooker.bookProfile("dxy_vs_phi",
                                  "d_{xy} vs #phi;#phi [rad];#LTd_{xy}#GT [#mum]",
                                  50,
                                  -std::numbers::pi,
                                  std::numbers::pi,
                                  -0.01 * cmToUm,
                                  0.01 * cmToUm,
                                  "");
  p_dz_eta = ibooker.bookProfile(
      "dz_vs_eta", "d_{z} vs #eta;#eta;#LTd_{z}#GT [#mum]", 50, -3.0, 3.0, -0.05 * cmToUm, 0.05 * cmToUm, "");
  p_dz_phi = ibooker.bookProfile("dz_vs_phi",
                                 "d_{z} vs #phi;#phi [rad];#LTd_{z}#GT [#mum]",
                                 50,
                                 -std::numbers::pi,
                                 std::numbers::pi,
                                 -0.05 * cmToUm,
                                 0.05 * cmToUm,
                                 "");

  // 2D eta-phi occupancy histograms
  h2_eta_phi = ibooker.book2I(
      "eta_vs_phi", "Track occupancy;#eta;#phi [rad]", 50, -3.0, 3.0, 50, -std::numbers::pi, std::numbers::pi);

  // 2D eta-phi profiles
  p2_dxy_eta_phi = ibooker.bookProfile2D("dxy_vs_eta_phi",
                                         "d_{xy} vs #eta-#phi;#eta;#phi [rad];#LTd_{xy}#GT [#mum]",
                                         50,
                                         -3.0,
                                         3.0,
                                         50,
                                         -std::numbers::pi,
                                         std::numbers::pi,
                                         -0.01 * cmToUm,
                                         0.01 * cmToUm,
                                         "");

  p2_dz_eta_phi = ibooker.bookProfile2D("dz_vs_eta_phi",
                                        "d_{z} vs #eta-#phi;#eta;#phi [rad];#LTd_{z}#GT [#mum]",
                                        50,
                                        -3.0,
                                        3.0,
                                        50,
                                        -std::numbers::pi,
                                        std::numbers::pi,
                                        -0.05 * cmToUm,
                                        0.05 * cmToUm,
                                        "");

  p2_nValidPixelHits_eta_phi =
      ibooker.bookProfile2D("nValidPixelHits_vs_eta_phi_prof",
                            "nValidPixelHits vs #eta-#phi;#eta;#phi [rad];#LTnValidPixelHits#GT",
                            50,
                            -3.0,
                            3.0,
                            50,
                            -std::numbers::pi,
                            std::numbers::pi,
                            0.,
                            10.,
                            "");

  p2_nTrackerLayersWithMeasurement_eta_phi = ibooker.bookProfile2D(
      "nTrackerLayersWithMeasurement_vs_eta_phi_prof",
      "nTrackerLayersWithMeasurement vs #eta-#phi;#eta;#phi [rad];#LTnTrackerLayersWithMeasurement#GT",
      50,
      -3.0,
      3.0,
      50,
      -std::numbers::pi,
      std::numbers::pi,
      0.,
      20.,
      "");

  p2_nValidStripHits_eta_phi =
      ibooker.bookProfile2D("nValidStripHits_vs_eta_phi_prof",
                            "nValidStripHits vs #eta-#phi;#eta;#phi [rad];#LTnValidStripHits#GT",
                            50,
                            -3.0,
                            3.0,
                            50,
                            -std::numbers::pi,
                            std::numbers::pi,
                            0.,
                            30.,
                            "");
}

// main event loop
void ScoutingTrackMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  auto const& tracks = iEvent.get(tracksToken_);
  auto const& vertices = iEvent.get(verticesToken_);

  if (vertices.empty())
    return;

  for (const auto& trk : tracks) {
    // --- build reco track ---
    reco::Track recoTrk = makeRecoTrack(trk);

    auto [vtxIndex, closestVtx] = findClosestScoutingVertex(&recoTrk, vertices);
    if (!closestVtx)
      continue;

    h_vtx_idx->Fill(vtxIndex);

    const float eta = recoTrk.eta();
    const float phi = recoTrk.phi();

    // --- fill 2D eta-phi occupancy histograms ---
    h2_eta_phi->Fill(eta, phi);
    p2_nValidPixelHits_eta_phi->Fill(eta, phi, trk.tk_nValidPixelHits());
    p2_nTrackerLayersWithMeasurement_eta_phi->Fill(eta, phi, trk.tk_nTrackerLayersWithMeasurement());
    p2_nValidStripHits_eta_phi->Fill(eta, phi, trk.tk_nValidStripHits());

    // --- build reco vertex ---
    reco::Vertex recoVtx = makeRecoVertex(*closestVtx);

    // --- impact parameters (standard CMSSW definitions) ---
    float dxy = recoTrk.dxy(recoVtx.position());
    float dz = recoTrk.dz(recoVtx.position());

    if (recoTrk.pt() < 3.)
      continue;

    // --- fill histograms ---
    h_dxy->Fill(dxy * cmToUm);
    h_dz->Fill(dz * cmToUm);

    p_dxy_eta->Fill(recoTrk.eta(), dxy * cmToUm);
    p_dxy_phi->Fill(recoTrk.phi(), dxy * cmToUm);

    p_dz_eta->Fill(recoTrk.eta(), dz * cmToUm);
    p_dz_phi->Fill(recoTrk.phi(), dz * cmToUm);

    // --- fill 2D eta-phi profiles ---
    p2_dxy_eta_phi->Fill(eta, phi, dxy * cmToUm);
    p2_dz_eta_phi->Fill(eta, phi, dz * cmToUm);
  }
}

// helper: build reco::Track
reco::Track ScoutingTrackMonitor::makeRecoTrack(const Run3ScoutingTrack& sTrack) const {
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

// helper: build reco::Vertex
reco::Vertex ScoutingTrackMonitor::makeRecoVertex(const Run3ScoutingVertex& sVertex) const {
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

std::pair<unsigned int, const Run3ScoutingVertex*> ScoutingTrackMonitor::findClosestScoutingVertex(
    const reco::Track* track, const std::vector<Run3ScoutingVertex>& vertices) {
  double minDistance = std::numeric_limits<double>::max();
  const Run3ScoutingVertex* closestVertex = nullptr;

  unsigned int index{0}, theIndex{999};

  for (const auto& vertex : vertices) {
    math::XYZPoint vertexPosition(vertex.x(), vertex.y(), vertex.z());

    const auto& trackMomentum = track->momentum();
    const auto& vertexToPoint = vertexPosition - track->referencePoint();

    double distance = vertexToPoint.Cross(trackMomentum).R() / trackMomentum.R();

    if (distance < minDistance) {
      minDistance = distance;
      closestVertex = &vertex;
      theIndex = index;
    }
    index++;
  }
  return std::make_pair(theIndex, closestVertex);
}

void ScoutingTrackMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracks", edm::InputTag("hltScoutingTrackPacker"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"));
  desc.add<std::string>("topFolderName", "HLT/ScoutingOffline/Tracks");
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ScoutingTrackMonitor);
