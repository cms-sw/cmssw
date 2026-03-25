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

  h_dxy = ibooker.book1D("dxy", "dxy;dxy [cm];Entries", 100, -0.5, 0.5);
  h_dz = ibooker.book1D("dz", "dz;dz [cm];Entries", 100, -1.0, 1.0);

  h_vtx_idx = ibooker.book1DD("vertexIndex", "tracks Vertex Index;Vertex index;Entries", 17, -1.5, 15.5);

  p_dxy_eta = ibooker.bookProfile("dxy_vs_eta", "dxy vs eta;eta;<dxy>", 50, -3.0, 3.0, -0.5, 0.5, "");
  p_dxy_phi =
      ibooker.bookProfile("dxy_vs_phi", "dxy vs phi;phi;<dxy>", 50, -std::numbers::pi, std::numbers::pi, -0.5, 0.5, "");
  p_dz_eta = ibooker.bookProfile("dz_vs_eta", "dz vs eta;eta;<dz>", 50, -3.0, 3.0, -1.0, 1.0, "");
  p_dz_phi =
      ibooker.bookProfile("dz_vs_phi", "dz vs phi;phi;<dz>", 50, -std::numbers::pi, std::numbers::pi, -1.0, 1.0, "");
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

    // // --- find closest vertex in z ---
    // float bestDist = 1e9;
    // const Run3ScoutingVertex* closestVtx = nullptr;
    // for (const auto& vtx : vertices) {
    //   float dz = std::abs(trk.tk_vz() - vtx.z());
    //   if (dz < bestDist) {
    //     bestDist = dz;
    //     closestVtx = &vtx;
    //   }
    // }

    // if (!closestVtx)
    //   continue;

    // --- build reco vertex ---
    reco::Vertex recoVtx = makeRecoVertex(*closestVtx);

    // --- impact parameters (standard CMSSW definitions) ---
    float dxy = recoTrk.dxy(recoVtx.position());
    float dz = recoTrk.dz(recoVtx.position());

    if (recoTrk.pt() < 3.)
      continue;

    // --- fill histograms ---
    h_dxy->Fill(dxy);
    h_dz->Fill(dz);

    p_dxy_eta->Fill(recoTrk.eta(), dxy);
    p_dxy_phi->Fill(recoTrk.phi(), dxy);

    p_dz_eta->Fill(recoTrk.eta(), dz);
    p_dz_phi->Fill(recoTrk.phi(), dz);
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
