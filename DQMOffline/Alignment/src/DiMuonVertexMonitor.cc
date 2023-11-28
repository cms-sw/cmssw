/*
 *  See header file for a description of this class.
 *
 */

#include <fmt/printf.h>

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
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

#include "TLorentzVector.h"

namespace {
  constexpr float cmToum = 10e4;
  constexpr float mumass2 = 0.105658367 * 0.105658367;  //mu mass squared (GeV^2/c^4)
}  // namespace

DiMuonVertexMonitor::DiMuonVertexMonitor(const edm::ParameterSet& iConfig)
    : ttbESToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("muonTracks"))),
      vertexToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      motherName_(iConfig.getParameter<std::string>("decayMotherName")),
      MEFolderName_(iConfig.getParameter<std::string>("FolderName")),
      useClosestVertex_(iConfig.getParameter<bool>("useClosestVertex")),
      maxSVdist_(iConfig.getParameter<double>("maxSVdist")),
      CosPhi3DConfiguration_(iConfig.getParameter<edm::ParameterSet>("CosPhi3DConfig")),
      SVDistConfiguration_(iConfig.getParameter<edm::ParameterSet>("SVDistConfig")),
      SVDistSigConfiguration_(iConfig.getParameter<edm::ParameterSet>("SVDistSigConfig")),
      SVDist3DConfiguration_(iConfig.getParameter<edm::ParameterSet>("SVDist3DConfig")),
      SVDist3DSigConfiguration_(iConfig.getParameter<edm::ParameterSet>("SVDist3DSigConfig")) {
  if (motherName_.find('Z') != std::string::npos) {
    massLimits_ = std::make_pair(50., 120);
  } else if (motherName_.find("J/#psi") != std::string::npos) {
    massLimits_ = std::make_pair(2.7, 3.4);
  } else if (motherName_.find("#Upsilon") != std::string::npos) {
    massLimits_ = std::make_pair(8.9, 9.9);
  } else {
    edm::LogError("DiMuonVertexMonitor") << " unrecognized decay mother particle: " << motherName_
                                         << " setting the default for the Z->mm (50.,120.)" << std::endl;
    massLimits_ = std::make_pair(50., 120);
  }
}

void DiMuonVertexMonitor::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  iBooker.setCurrentFolder(MEFolderName_ + "/DiMuonVertexMonitor");

  // clang-format off
  std::string ts = fmt::sprintf(";%s vertex probability;N(#mu#mu pairs)", motherName_);
  std::string ps = "N(#mu#mu pairs)";
  hSVProb_ = iBooker.book1D("VtxProb", ts, 100, 0., 1.);

  ts = fmt::sprintf("#chi^{2} of the %s vertex; #chi^{2} of the %s vertex; %s", motherName_, motherName_, ps);
  hSVChi2_ = iBooker.book1D("VtxChi2", ts, 200, 0., 200.);

  ts = fmt::sprintf("#chi^{2}/ndf of the %s vertex; #chi^{2}/ndf of %s vertex; %s", motherName_, motherName_, ps);
  hSVNormChi2_ = iBooker.book1D("VtxNormChi2", ts, 100, 0., 20.);

  std::string histTit = motherName_ + " #rightarrow #mu^{+}#mu^{-}";
  ts = fmt::sprintf("%s;PV- %sV xy distance [#mum];%s", histTit, motherName_, ps);
  hSVDist_ = iBooker.book1D("VtxDist", ts, 100, 0., 300.);

  ts = fmt::sprintf("%s;PV-%sV xy distance error [#mum];%s", histTit, motherName_, ps);
  hSVDistErr_ = iBooker.book1D("VtxDistErr", ts, 100, 0., 1000.);

  ts = fmt::sprintf("%s;PV-%sV xy distance signficance;%s", histTit, motherName_, ps);
  hSVDistSig_ = iBooker.book1D("VtxDistSig", ts, 100, 0., 5.);

  ts = fmt::sprintf("compatibility of %s vertex; compatibility of %s vertex; %s", motherName_, motherName_, ps);
  hSVCompatibility_ = iBooker.book1D("VtxCompatibility", ts, 100, 0., 100.);

  ts = fmt::sprintf("%s;PV-%sV 3D distance [#mum];%s", histTit, motherName_, ps);
  hSVDist3D_ = iBooker.book1D("VtxDist3D", ts, 100, 0., 300.);

  ts = fmt::sprintf("%s;PV-%sV 3D distance error [#mum];%s", histTit, motherName_, ps);
  hSVDist3DErr_ = iBooker.book1D("VtxDist3DErr", ts, 100, 0., 1000.);

  ts = fmt::sprintf("%s;PV-%sV 3D distance signficance;%s", histTit, motherName_, ps);
  hSVDist3DSig_ = iBooker.book1D("VtxDist3DSig", ts, 100, 0., 5.);

  ts = fmt::sprintf("3D compatibility of %s vertex;3D compatibility of %s vertex; %s", motherName_, motherName_, ps);
  hSVCompatibility3D_ = iBooker.book1D("VtxCompatibility3D", ts, 100, 0., 100.);

  hInvMass_ = iBooker.book1D("InvMass", fmt::sprintf("%s;M(#mu,#mu) [GeV];%s", histTit, ps), 70., massLimits_.first, massLimits_.second);
  hCosPhi_ = iBooker.book1D("CosPhi", fmt::sprintf("%s;cos(#phi_{xy});%s", histTit, ps), 50, -1., 1.);
  hCosPhi3D_ = iBooker.book1D("CosPhi3D", fmt::sprintf("%s;cos(#phi_{3D});%s", histTit, ps), 50, -1., 1.);
  hCosPhiInv_ = iBooker.book1D("CosPhiInv", fmt::sprintf("%s;inverted cos(#phi_{xy});%s", histTit, ps), 50, -1., 1.);
  hCosPhiInv3D_ = iBooker.book1D("CosPhiInv3D", fmt::sprintf("%s;inverted cos(#phi_{3D});%s", histTit, ps), 50, -1., 1.);
  hCosPhiUnbalance_ = iBooker.book1D("CosPhiUnbalance", fmt::sprintf("%s;cos(#phi_{xy}) unbalance;#Delta%s", histTit, ps), 50, -1.,1.);
  hCosPhi3DUnbalance_ = iBooker.book1D("CosPhi3DUnbalance", fmt::sprintf("%s;cos(#phi_{3D}) unbalance;#Delta%s", histTit, ps), 50, -1., 1.);

  hdxy_ = iBooker.book1D("dxy", fmt::sprintf("%s;muon track d_{xy}(PV) [#mum];muon tracks", histTit), 150, -300, 300);
  hdz_ = iBooker.book1D("dz", fmt::sprintf("%s;muon track d_{z}(PV) [#mum];muon tracks", histTit), 150, -300, 300);
  hdxyErr_ = iBooker.book1D("dxyErr", fmt::sprintf("%s;muon track err_{dxy} [#mum];muon tracks", histTit), 250, 0., 500.);
  hdzErr_ = iBooker.book1D("dzErr", fmt::sprintf("%s;muon track err_{dz} [#mum];muon tracks", histTit), 250, 0., 500.);
  hIP2d_ = iBooker.book1D("IP2d", fmt::sprintf("%s;muon track IP_{2D} [#mum];muon tracks", histTit), 150, -300, 300);
  hIP3d_ = iBooker.book1D("IP3d", fmt::sprintf("%s;muon track IP_{3D} [#mum];muon tracks", histTit), 150, -300, 300);
  hIP2dsig_ = iBooker.book1D("IP2Dsig", fmt::sprintf("%s;muon track IP_{2D} significance;muon tracks", histTit), 100, 0., 5.);
  hIP3dsig_ = iBooker.book1D("IP3Dsig", fmt::sprintf("%s;muon track IP_{3D} significance;muon tracks", histTit), 100, 0., 5.);
  // clang-format on

  // now book the cosphi3D plots vs kinematics
  iBooker.setCurrentFolder(MEFolderName_ + "/DiMuonVertexMonitor/CosPhi3DPlots");
  CosPhi3DPlots_.bookFromPSet(iBooker, CosPhi3DConfiguration_);

  // now book the PV-SV distance plots vs kinematics
  iBooker.setCurrentFolder(MEFolderName_ + "/DiMuonVertexMonitor/SVDistPlots");
  SVDistPlots_.bookFromPSet(iBooker, SVDistConfiguration_);

  // now book the PV-SV distance significance plots vs kinematics
  iBooker.setCurrentFolder(MEFolderName_ + "/DiMuonVertexMonitor/SVDistSigPlots");
  SVDistSigPlots_.bookFromPSet(iBooker, SVDistSigConfiguration_);

  // now book the PV-SV 3D distance plots vs kinematics
  iBooker.setCurrentFolder(MEFolderName_ + "/DiMuonVertexMonitor/SVDist3DPlots");
  SVDist3DPlots_.bookFromPSet(iBooker, SVDist3DConfiguration_);

  // now book the PV-SV 3D distance significance plots vs kinematics
  iBooker.setCurrentFolder(MEFolderName_ + "/DiMuonVertexMonitor/SVDist3DSigPlots");
  SVDist3DSigPlots_.bookFromPSet(iBooker, SVDist3DSigConfiguration_);
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

#ifdef EDM_ML_DEBUG
  for (const auto& track : myTracks) {
    edm::LogVerbatim("DiMuonVertexMonitor") << __PRETTY_FUNCTION__ << " pT: " << track->pt() << " GeV"
                                            << " , pT error: " << track->ptError() << " GeV"
                                            << " , eta: " << track->eta() << " , phi: " << track->phi() << std::endl;
  }
#endif

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

#ifdef EDM_ML_DEBUG
  // Define a lambda function to convert TLorentzVector to a string
  auto tLorentzVectorToString = [](const TLorentzVector& vector) {
    return std::to_string(vector.Px()) + " " + std::to_string(vector.Py()) + " " + std::to_string(vector.Pz()) + " " +
           std::to_string(vector.E());
  };

  edm::LogVerbatim("DiMuonVertexMonitor") << "mu+" << tLorentzVectorToString(p4_tplus) << std::endl;
  edm::LogVerbatim("DiMuonVertexMonitor") << "mu-" << tLorentzVectorToString(p4_tminus) << std::endl;
#endif

  const auto& Zp4 = p4_tplus + p4_tminus;
  float track_invMass = Zp4.M();
  hInvMass_->Fill(track_invMass);

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
  hSVChi2_->Fill(mumuTransientVtx.totalChiSquared());
  hSVNormChi2_->Fill(mumuTransientVtx.totalChiSquared() / (int)mumuTransientVtx.degreesOfFreedom());

  if (!mumuTransientVtx.isValid())
    return;

  const reco::Vertex* theClosestVertex;
  // get collection of reconstructed vertices from event
  edm::Handle<reco::VertexCollection> vertexHandle = iEvent.getHandle(vertexToken_);
  if (vertexHandle.isValid()) {
    const reco::VertexCollection* vertices = vertexHandle.product();
    theClosestVertex = this->findClosestVertex(mumuTransientVtx, vertices);
  } else {
    edm::LogWarning("DiMuonVertexMonitor") << "invalid vertex collection encountered Skipping event!";
    return;
  }

  reco::Vertex theMainVtx;
  if (!useClosestVertex_ || theClosestVertex == nullptr) {
    // if the closest vertex is not available, or explicitly not chosen
    theMainVtx = vertexHandle.product()->front();
  } else {
    theMainVtx = *theClosestVertex;
  }

  const math::XYZPoint theMainVtxPos(theMainVtx.position().x(), theMainVtx.position().y(), theMainVtx.position().z());
  const math::XYZPoint myVertex(
      mumuTransientVtx.position().x(), mumuTransientVtx.position().y(), mumuTransientVtx.position().z());
  const math::XYZPoint deltaVtx(
      theMainVtxPos.x() - myVertex.x(), theMainVtxPos.y() - myVertex.y(), theMainVtxPos.z() - myVertex.z());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DiMuonVertexMonitor") << "mm vertex position:" << mumuTransientVtx.position().x() << ","
                                          << mumuTransientVtx.position().y() << "," << mumuTransientVtx.position().z();

  edm::LogVerbatim("DiMuonVertexMonitor") << "main vertex position:" << theMainVtx.position().x() << ","
                                          << theMainVtx.position().y() << "," << theMainVtx.position().z();
#endif

  if (theMainVtx.isValid()) {
    // fill the impact parameter plots
    for (const auto& track : myTracks) {
      hdxy_->Fill(track->dxy(theMainVtxPos) * cmToum);
      hdz_->Fill(track->dz(theMainVtxPos) * cmToum);
      hdxyErr_->Fill(track->dxyError() * cmToum);
      hdzErr_->Fill(track->dzError() * cmToum);

      const auto& ttrk = theB->build(track);
      Global3DVector dir(track->px(), track->py(), track->pz());
      const auto& ip2d = IPTools::signedTransverseImpactParameter(ttrk, dir, theMainVtx);
      const auto& ip3d = IPTools::signedImpactParameter3D(ttrk, dir, theMainVtx);

      hIP2d_->Fill(ip2d.second.value() * cmToum);
      hIP3d_->Fill(ip3d.second.value() * cmToum);
      hIP2dsig_->Fill(ip2d.second.significance());
      hIP3dsig_->Fill(ip3d.second.significance());
    }

    // Z Vertex distance in the xy plane
    VertexDistanceXY vertTool;
    double distance = vertTool.distance(mumuTransientVtx, theMainVtx).value();
    double dist_err = vertTool.distance(mumuTransientVtx, theMainVtx).error();
    float compatibility = 0.;

    try {
      compatibility = vertTool.compatibility(mumuTransientVtx, theMainVtx);
    } catch (cms::Exception& er) {
      LogTrace("DiMuonVertexMonitor") << "caught std::exception " << er.what() << std::endl;
    }

    hSVCompatibility_->Fill(compatibility);
    hSVDist_->Fill(distance * cmToum);
    hSVDistErr_->Fill(dist_err * cmToum);
    hSVDistSig_->Fill(distance / dist_err);

    // Z Vertex distance in 3D
    VertexDistance3D vertTool3D;
    double distance3D = vertTool3D.distance(mumuTransientVtx, theMainVtx).value();
    double dist3D_err = vertTool3D.distance(mumuTransientVtx, theMainVtx).error();
    float compatibility3D = 0.;

    try {
      compatibility3D = vertTool3D.compatibility(mumuTransientVtx, theMainVtx);
    } catch (cms::Exception& er) {
      LogTrace("DiMuonVertexMonitor") << "caught std::exception " << er.what() << std::endl;
    }

    hSVCompatibility3D_->Fill(compatibility3D);
    hSVDist3D_->Fill(distance3D * cmToum);
    hSVDist3DErr_->Fill(dist3D_err * cmToum);
    hSVDist3DSig_->Fill(distance3D / dist3D_err);

    // creat the pair of TLorentVectors used to make the plos
    std::pair<TLorentzVector, TLorentzVector> tktk_p4 = std::make_pair(p4_tplus, p4_tminus);

    SVDistPlots_.fillPlots(distance * cmToum, tktk_p4);
    SVDistSigPlots_.fillPlots(distance / dist_err, tktk_p4);
    SVDist3DPlots_.fillPlots(distance3D * cmToum, tktk_p4);
    SVDist3DSigPlots_.fillPlots(distance3D / dist3D_err, tktk_p4);

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

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DiMuonVertexMonitor")
          << "distance " << distance * cmToum << " cosphi3D:" << cosphi3D << std::endl;
#endif

      // unbalance
      hCosPhiUnbalance_->Fill(cosphi, 1.);
      hCosPhiUnbalance_->Fill(-cosphi, -1.);
      hCosPhi3DUnbalance_->Fill(cosphi3D, 1.);
      hCosPhi3DUnbalance_->Fill(-cosphi3D, -1.);

      // fill the cos(phi3D) plots
      CosPhi3DPlots_.fillPlots(cosphi3D, tktk_p4);
    }
  } else {
    edm::LogWarning("DiMuonVertexMonitor") << "hardest primary vertex in the event is not valid!";
  }
}

// compute the closest vertex to di-lepton ------------------------------------
const reco::Vertex* DiMuonVertexMonitor::findClosestVertex(const TransientVertex aTransVtx,
                                                           const reco::VertexCollection* vertices) const {
  reco::Vertex* defaultVtx = nullptr;

  if (!aTransVtx.isValid())
    return defaultVtx;

  // find the closest vertex to the secondary vertex in 3D
  VertexDistance3D vertTool3D;
  float minD = 9999.;
  int closestVtxIndex = 0;
  int counter = 0;
  for (const auto& vtx : *vertices) {
    double dist3D = vertTool3D.distance(aTransVtx, vtx).value();
    if (dist3D < minD) {
      minD = dist3D;
      closestVtxIndex = counter;
    }
    counter++;
  }

  if ((*vertices).at(closestVtxIndex).isValid()) {
    return &(vertices->at(closestVtxIndex));
  } else {
    return defaultVtx;
  }
}

void DiMuonVertexMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonTracks", edm::InputTag("ALCARECOTkAlDiMuon"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<std::string>("FolderName", "DiMuonVertexMonitor");
  desc.add<std::string>("decayMotherName", "Z");
  desc.add<bool>("useClosestVertex", true);
  desc.add<double>("maxSVdist", 50.);

  {
    edm::ParameterSetDescription psCosPhi3D;
    psCosPhi3D.add<std::string>("name", "CosPhi3D");
    psCosPhi3D.add<std::string>("title", "cos(#phi_{3D})");
    psCosPhi3D.add<std::string>("yUnits", "");
    psCosPhi3D.add<int>("NxBins", 24);
    psCosPhi3D.add<int>("NyBins", 50);
    psCosPhi3D.add<double>("ymin", -1.);
    psCosPhi3D.add<double>("ymax", 1.);
    psCosPhi3D.add<double>("maxDeltaEta", 3.7);
    desc.add<edm::ParameterSetDescription>("CosPhi3DConfig", psCosPhi3D);
  }

  {
    edm::ParameterSetDescription psSVDist;
    psSVDist.add<std::string>("name", "SVDist");
    psSVDist.add<std::string>("title", "PV-SV distance");
    psSVDist.add<std::string>("yUnits", "[#mum]");
    psSVDist.add<int>("NxBins", 24);
    psSVDist.add<int>("NyBins", 100);
    psSVDist.add<double>("ymin", 0.);
    psSVDist.add<double>("ymax", 300.);
    psSVDist.add<double>("maxDeltaEta", 3.7);
    desc.add<edm::ParameterSetDescription>("SVDistConfig", psSVDist);
  }

  {
    edm::ParameterSetDescription psSVDistSig;
    psSVDistSig.add<std::string>("name", "SVDistSig");
    psSVDistSig.add<std::string>("title", "PV-SV distance significance");
    psSVDistSig.add<std::string>("yUnits", "[#mum]");
    psSVDistSig.add<int>("NxBins", 24);
    psSVDistSig.add<int>("NyBins", 100);
    psSVDistSig.add<double>("ymin", 0.);
    psSVDistSig.add<double>("ymax", 5.);
    psSVDistSig.add<double>("maxDeltaEta", 3.7);
    desc.add<edm::ParameterSetDescription>("SVDistSigConfig", psSVDistSig);
  }

  {
    edm::ParameterSetDescription psSVDist3D;
    psSVDist3D.add<std::string>("name", "SVDist3D");
    psSVDist3D.add<std::string>("title", "PV-SV 3D distance");
    psSVDist3D.add<std::string>("yUnits", "[#mum]");
    psSVDist3D.add<int>("NxBins", 24);
    psSVDist3D.add<int>("NyBins", 100);
    psSVDist3D.add<double>("ymin", 0.);
    psSVDist3D.add<double>("ymax", 300.);
    psSVDist3D.add<double>("maxDeltaEta", 3.7);
    desc.add<edm::ParameterSetDescription>("SVDist3DConfig", psSVDist3D);
  }

  {
    edm::ParameterSetDescription psSVDist3DSig;
    psSVDist3DSig.add<std::string>("name", "SVDist3DSig");
    psSVDist3DSig.add<std::string>("title", "PV-SV 3D distance significance");
    psSVDist3DSig.add<std::string>("yUnits", "[#mum]");
    psSVDist3DSig.add<int>("NxBins", 24);
    psSVDist3DSig.add<int>("NyBins", 100);
    psSVDist3DSig.add<double>("ymin", 0.);
    psSVDist3DSig.add<double>("ymax", 5.);
    psSVDist3DSig.add<double>("maxDeltaEta", 3.7);
    desc.add<edm::ParameterSetDescription>("SVDist3DSigConfig", psSVDist3DSig);
  }

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(DiMuonVertexMonitor);
