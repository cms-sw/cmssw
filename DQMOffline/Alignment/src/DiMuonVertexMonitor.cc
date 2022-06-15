/*
 *  See header file for a description of this class.
 *
 */

#include "DQMOffline/Alignment/interface/DiMuonVertexMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

#include "TLorentzVector.h"

namespace {
  constexpr float cmToum = 10e4;
  constexpr float mumass2 = 0.105658367 * 0.105658367;  //mu mass squared (GeV^2/c^4)
}  // namespace

DiMuonVertexMonitor::DiMuonVertexMonitor(const edm::ParameterSet& iConfig)
    : ttbESToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("muonTracks"))),
      vertexToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      MEFolderName_(iConfig.getParameter<std::string>("FolderName")),
      maxSVdist_(iConfig.getParameter<double>("maxSVdist")) {}

void DiMuonVertexMonitor::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  iBooker.setCurrentFolder(MEFolderName_ + "/DiMuonVertexMonitor");
  hSVProb_ = iBooker.book1D("VtxProb", ";ZV vertex probability;N(#mu#mu pairs)", 100, 0., 1.);
  hSVDist_ = iBooker.book1D("VtxDist", ";PV-ZV xy distance [#mum];N(#mu#mu pairs)", 100, 0., 300.);
  hSVDistErr_ = iBooker.book1D("VtxDistErr", ";PV-ZV xy distance error [#mum];N(#mu#mu pairs)", 100, 0., 1000.);
  hSVDistSig_ = iBooker.book1D("VtxDistSig", ";PV-ZV xy distance signficance;N(#mu#mu pairs)", 100, 0., 5.);
  hSVDist3D_ = iBooker.book1D("VtxDist3D", ";PV-ZV 3D distance [#mum];N(#mu#mu pairs)", 100, 0., 300.);
  hSVDist3DErr_ = iBooker.book1D("VtxDist3DErr", ";PV-ZV 3D distance error [#mum];N(#mu#mu pairs)", 100, 0., 1000.);
  hSVDist3DSig_ = iBooker.book1D("VtxDist3DSig", ";PV-ZV 3D distance signficance;N(#mu#mu pairs)", 100, 0., 5.);
  hTrackInvMass_ = iBooker.book1D("TkTkInvMass", ";M(tk,tk) [GeV];N(tk tk pairs)", 70., 50., 120.);
  hCosPhi_ = iBooker.book1D("CosPhi", ";cos(#phi_{xy});N(#mu#mu pairs)", 50, -1., 1.);
  hCosPhi3D_ = iBooker.book1D("CosPhi3D", ";cos(#phi_{3D});N(#mu#mu pairs)", 50, -1., 1.);
  hCosPhiInv_ = iBooker.book1D("CosPhiInv", ";inverted cos(#phi_{xy});N(#mu#mu pairs)", 50, -1., 1.);
  hCosPhiInv3D_ = iBooker.book1D("CosPhiInv3D", ";inverted cos(#phi_{3D});N(#mu#mu pairs)", 50, -1., 1.);
}

void DiMuonVertexMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::vector<const reco::Track*> myTracks;
  const auto trackHandle = iEvent.getHandle(tracksToken_);
  if (!trackHandle.isValid()) {
    edm::LogError("DiMuonVertexMonitor") << "invalid track collection encountered!";
    return;
  }

  for (const auto& muonTrk : *trackHandle) {
    myTracks.emplace_back(&muonTrk);
  }

  const TransientTrackBuilder* theB = &iSetup.getData(ttbESToken_);
  TransientVertex mumuTransientVtx;
  std::vector<reco::TransientTrack> tks;

  if (myTracks.size() != 2) {
    edm::LogWarning("DiMuonVertexMonitor") << "There are not enough tracks to monitor!";
    return;
  }

  const auto& t1 = myTracks[1]->momentum();
  const auto& t0 = myTracks[0]->momentum();
  const auto& ditrack = t1 + t0;

  const auto& tplus = myTracks[0]->charge() > 0 ? myTracks[0] : myTracks[1];
  const auto& tminus = myTracks[0]->charge() < 0 ? myTracks[0] : myTracks[1];

  TLorentzVector p4_tplus(tplus->px(), tplus->py(), tplus->pz(), sqrt((tplus->p() * tplus->p()) + mumass2));
  TLorentzVector p4_tminus(tminus->px(), tminus->py(), tminus->pz(), sqrt((tminus->p() * tminus->p()) + mumass2));

  const auto& Zp4 = p4_tplus + p4_tminus;
  float track_invMass = Zp4.M();
  hTrackInvMass_->Fill(track_invMass);

  // creat the pair of TLorentVectors used to make the plos
  std::pair<TLorentzVector, TLorentzVector> tktk_p4 = std::make_pair(p4_tplus, p4_tminus);

  math::XYZPoint ZpT(ditrack.x(), ditrack.y(), 0);
  math::XYZPoint Zp(ditrack.x(), ditrack.y(), ditrack.z());

  for (const auto& track : myTracks) {
    reco::TransientTrack trajectory = theB->build(track);
    tks.push_back(trajectory);
  }

  KalmanVertexFitter kalman(true);
  mumuTransientVtx = kalman.vertex(tks);

  double SVProb = TMath::Prob(mumuTransientVtx.totalChiSquared(), (int)mumuTransientVtx.degreesOfFreedom());
  hSVProb_->Fill(SVProb);

  if (!mumuTransientVtx.isValid())
    return;

  // get collection of reconstructed vertices from event
  edm::Handle<reco::VertexCollection> vertexHandle = iEvent.getHandle(vertexToken_);

  math::XYZPoint theMainVtxPos(0, 0, 0);
  reco::Vertex theMainVertex = vertexHandle.product()->front();

  if (vertexHandle.isValid()) {
    const reco::VertexCollection* vertices = vertexHandle.product();
    if ((*vertices)[0].isValid()) {
      auto theMainVtx = (*vertices)[0];
      theMainVtxPos.SetXYZ(theMainVtx.position().x(), theMainVtx.position().y(), theMainVtx.position().z());
    } else {
      edm::LogWarning("DiMuonVertexMonitor") << "hardest primary vertex in the event is not valid!";
    }
  } else {
    edm::LogWarning("DiMuonVertexMonitor") << "invalid vertex collection encountered!";
  }

  const math::XYZPoint myVertex(
      mumuTransientVtx.position().x(), mumuTransientVtx.position().y(), mumuTransientVtx.position().z());
  const math::XYZPoint deltaVtx(
      theMainVtxPos.x() - myVertex.x(), theMainVtxPos.y() - myVertex.y(), theMainVtxPos.z() - myVertex.z());

  if (theMainVertex.isValid()) {
    // Z Vertex distance in the xy plane

    VertexDistanceXY vertTool;
    double distance = vertTool.distance(mumuTransientVtx, theMainVertex).value();
    double dist_err = vertTool.distance(mumuTransientVtx, theMainVertex).error();

    hSVDist_->Fill(distance * cmToum);
    hSVDistErr_->Fill(dist_err * cmToum);
    hSVDistSig_->Fill(distance / dist_err);

    // Z Vertex distance in 3D
    VertexDistance3D vertTool3D;
    double distance3D = vertTool3D.distance(mumuTransientVtx, theMainVertex).value();
    double dist3D_err = vertTool3D.distance(mumuTransientVtx, theMainVertex).error();

    hSVDist3D_->Fill(distance3D * cmToum);
    hSVDist3DErr_->Fill(dist3D_err * cmToum);
    hSVDist3DSig_->Fill(distance3D / dist3D_err);

    // cut on the PV - SV distance
    if (distance * cmToum < maxSVdist_) {
      double cosphi = (ZpT.x() * deltaVtx.x() + ZpT.y() * deltaVtx.y()) /
                      (sqrt(ZpT.x() * ZpT.x() + ZpT.y() * ZpT.y()) *
                       sqrt(deltaVtx.x() * deltaVtx.x() + deltaVtx.y() * deltaVtx.y()));

      double cosphi3D = (Zp.x() * deltaVtx.x() + Zp.y() * deltaVtx.y() + Zp.z() * deltaVtx.z()) /
                        (sqrt(Zp.x() * Zp.x() + Zp.y() * Zp.y() + Zp.z() * Zp.z()) *
                         sqrt(deltaVtx.x() * deltaVtx.x() + deltaVtx.y() * deltaVtx.y() + deltaVtx.z() * deltaVtx.z()));

      hCosPhi_->Fill(cosphi);
      hCosPhi3D_->Fill(cosphi3D);
      // inverted
      hCosPhiInv_->Fill(-cosphi);
      hCosPhiInv3D_->Fill(-cosphi3D);
    }
  } else {
    edm::LogWarning("DiMuonVertexMonitor") << "hardest primary vertex in the event is not valid!";
  }
}

void DiMuonVertexMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonTracks", edm::InputTag("ALCARECOTkAlDiMuon"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<std::string>("FolderName", "DiMuonVertexMonitor");
  desc.add<double>("maxSVdist", 50.);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(DiMuonVertexMonitor);
