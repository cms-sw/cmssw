// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      DiMuonVertexValidation
//
/**\class DiMuonVertexValidation DiMuonVertexValidation.cc Alignment/OfflineValidation/plugins/DiMuonVertexValidation.cc

 Description: Class to perform validation Tracker Alignment Validations by means of a PV and the SV constructed with a di-muon pair

*/
//
// Original Author:  Marco Musich
//         Created:  Wed, 21 Apr 2021 09:06:25 GMT
//
//

// system include files
#include <memory>
#include <utility>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// muons
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

// utils
#include "Alignment/OfflineValidation/interface/DiLeptonVertexHelpers.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TLorentzVector.h"

// tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

// vertices
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// ROOT
#include "TH1F.h"
#include "TH2F.h"

//#define LogDebug(X) std::cout << X <<

//
// class declaration
//

class DiMuonVertexValidation : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit DiMuonVertexValidation(const edm::ParameterSet&);
  ~DiMuonVertexValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  DiLeptonHelp::Counts myCounts;

  bool useReco_;
  std::vector<double> pTthresholds_;
  float maxSVdist_;

  // plot configurations

  edm::ParameterSet CosPhiConfiguration_;
  edm::ParameterSet CosPhi3DConfiguration_;
  edm::ParameterSet VtxProbConfiguration_;
  edm::ParameterSet VtxDistConfiguration_;
  edm::ParameterSet VtxDist3DConfiguration_;
  edm::ParameterSet VtxDistSigConfiguration_;
  edm::ParameterSet VtxDist3DSigConfiguration_;
  edm::ParameterSet DiMuMassConfiguration_;

  // control plots

  TH1F* hSVProb_;
  TH1F* hSVDist_;
  TH1F* hSVDistSig_;
  TH1F* hSVDist3D_;
  TH1F* hSVDist3DSig_;

  TH1F* hCosPhi_;
  TH1F* hCosPhi3D_;
  TH1F* hCosPhiInv_;
  TH1F* hCosPhiInv3D_;

  TH1F* hInvMass_;
  TH1F* hTrackInvMass_;

  TH1F* hCutFlow_;

  // 2D maps

  DiLeptonHelp::PlotsVsKinematics CosPhiPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics CosPhi3DPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics VtxProbPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics VtxDistPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics VtxDist3DPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics VtxDistSigPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics VtxDist3DSigPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics ZMassPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);

  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbESToken_;

  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;   //used to select what tracks to read from configuration file
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;  //used to select what vertices to read from configuration file

  // either on or the other!
  edm::EDGetTokenT<reco::MuonCollection> muonsToken_;  //used to select what tracks to read from configuration file
  edm::EDGetTokenT<reco::TrackCollection>
      alcaRecoToken_;  //used to select what muon tracks to read from configuration file
};

//
// constants, enums and typedefs
//

static constexpr float cmToum = 10e4;
static constexpr float mumass2 = 0.105658367 * 0.105658367;  //mu mass squared (GeV^2/c^4)

//
// static data member definitions
//

//
// constructors and destructor
//
DiMuonVertexValidation::DiMuonVertexValidation(const edm::ParameterSet& iConfig)
    : useReco_(iConfig.getParameter<bool>("useReco")),
      pTthresholds_(iConfig.getParameter<std::vector<double>>("pTThresholds")),
      maxSVdist_(iConfig.getParameter<double>("maxSVdist")),
      CosPhiConfiguration_(iConfig.getParameter<edm::ParameterSet>("CosPhiConfig")),
      CosPhi3DConfiguration_(iConfig.getParameter<edm::ParameterSet>("CosPhi3DConfig")),
      VtxProbConfiguration_(iConfig.getParameter<edm::ParameterSet>("VtxProbConfig")),
      VtxDistConfiguration_(iConfig.getParameter<edm::ParameterSet>("VtxDistConfig")),
      VtxDist3DConfiguration_(iConfig.getParameter<edm::ParameterSet>("VtxDist3DConfig")),
      VtxDistSigConfiguration_(iConfig.getParameter<edm::ParameterSet>("VtxDistSigConfig")),
      VtxDist3DSigConfiguration_(iConfig.getParameter<edm::ParameterSet>("VtxDist3DSigConfig")),
      DiMuMassConfiguration_(iConfig.getParameter<edm::ParameterSet>("DiMuMassConfig")),
      ttbESToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
      vertexToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))) {
  if (useReco_) {
    muonsToken_ = mayConsume<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"));
  } else {
    alcaRecoToken_ = mayConsume<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("muonTracks"));
  }

  usesResource(TFileService::kSharedResource);

  // sort the vector of thresholds
  std::sort(pTthresholds_.begin(), pTthresholds_.end(), [](const double& lhs, const double& rhs) { return lhs > rhs; });

  edm::LogInfo("DiMuonVertexValidation") << __FUNCTION__;
  for (const auto& thr : pTthresholds_) {
    edm::LogInfo("DiMuonVertexValidation") << " Threshold: " << thr << " ";
  }
  edm::LogInfo("DiMuonVertexValidation") << "Max SV distance: " << maxSVdist_ << " ";
}

DiMuonVertexValidation::~DiMuonVertexValidation() = default;

//
// member functions
//

// ------------ method called for each event  ------------
void DiMuonVertexValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  myCounts.eventsTotal++;

  // the di-muon tracks
  std::vector<const reco::Track*> myTracks;

  // if we have to start from scratch from RECO data-tier
  if (useReco_) {
    // select the good muons
    std::vector<const reco::Muon*> myGoodMuonVector;
    for (const auto& muon : iEvent.get(muonsToken_)) {
      const reco::TrackRef t = muon.innerTrack();
      if (!t.isNull()) {
        if (t->quality(reco::TrackBase::highPurity)) {
          if (t->chi2() / t->ndof() <= 2.5 && t->numberOfValidHits() >= 5 &&
              t->hitPattern().numberOfValidPixelHits() >= 2 && t->quality(reco::TrackBase::highPurity))
            myGoodMuonVector.emplace_back(&muon);
        }
      }
    }

    LogDebug("DiMuonVertexValidation") << "myGoodMuonVector size: " << myGoodMuonVector.size() << std::endl;
    std::sort(myGoodMuonVector.begin(), myGoodMuonVector.end(), [](const reco::Muon*& lhs, const reco::Muon*& rhs) {
      return lhs->pt() > rhs->pt();
    });

    // just check the ordering
    for (const auto& muon : myGoodMuonVector) {
      LogDebug("DiMuonVertexValidation") << "pT: " << muon->pt() << " ";
    }
    LogDebug("DiMuonVertexValidation") << std::endl;

    // reject if there's no Z
    if (myGoodMuonVector.size() < 2)
      return;

    myCounts.eventsAfterMult++;

    if ((myGoodMuonVector[0]->pt()) < pTthresholds_[0] || (myGoodMuonVector[1]->pt() < pTthresholds_[1]))
      return;

    myCounts.eventsAfterPt++;
    myCounts.eventsAfterEta++;

    if (myGoodMuonVector[0]->charge() * myGoodMuonVector[1]->charge() > 0)
      return;

    const auto& m1 = myGoodMuonVector[1]->p4();
    const auto& m0 = myGoodMuonVector[0]->p4();
    const auto& mother = m0 + m1;

    float invMass = mother.M();
    hInvMass_->Fill(invMass);

    // just copy the top two muons
    std::vector<const reco::Muon*> theZMuonVector;
    theZMuonVector.reserve(2);
    theZMuonVector.emplace_back(myGoodMuonVector[1]);
    theZMuonVector.emplace_back(myGoodMuonVector[0]);

    // do the matching of Z muons with inner tracks
    unsigned int i = 0;
    for (const auto& muon : theZMuonVector) {
      i++;
      float minD = 1000.;
      const reco::Track* theMatch = nullptr;
      for (const auto& track : iEvent.get(tracksToken_)) {
        float D = ::deltaR(muon->eta(), muon->phi(), track.eta(), track.phi());
        if (D < minD) {
          minD = D;
          theMatch = &track;
        }
      }
      LogDebug("DiMuonVertexValidation") << "pushing new track: " << i << std::endl;
      myTracks.emplace_back(theMatch);
    }
  } else {
    // we start directly with the pre-selected ALCARECO tracks
    for (const auto& muon : iEvent.get(alcaRecoToken_)) {
      myTracks.emplace_back(&muon);
    }
  }

  LogDebug("DiMuonVertexValidation") << "selected tracks: " << myTracks.size() << std::endl;

  const TransientTrackBuilder* theB = &iSetup.getData(ttbESToken_);
  TransientVertex aTransientVertex;
  std::vector<reco::TransientTrack> tks;

  if (myTracks.size() != 2)
    return;

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

  // fill the z->mm mass plots
  ZMassPlots.fillPlots(track_invMass, tktk_p4);

  math::XYZPoint ZpT(ditrack.x(), ditrack.y(), 0);
  math::XYZPoint Zp(ditrack.x(), ditrack.y(), ditrack.z());

  for (const auto& track : myTracks) {
    reco::TransientTrack trajectory = theB->build(track);
    tks.push_back(trajectory);
  }

  KalmanVertexFitter kalman(true);
  aTransientVertex = kalman.vertex(tks);

  double SVProb = TMath::Prob(aTransientVertex.totalChiSquared(), (int)aTransientVertex.degreesOfFreedom());

  LogDebug("DiMuonVertexValidation") << " vertex prob: " << SVProb << std::endl;

  hSVProb_->Fill(SVProb);

  if (!aTransientVertex.isValid())
    return;

  myCounts.eventsAfterVtx++;

  // fill the VtxProb plots
  VtxProbPlots.fillPlots(SVProb, tktk_p4);

  // get collection of reconstructed vertices from event
  edm::Handle<reco::VertexCollection> vertexHandle = iEvent.getHandle(vertexToken_);

  math::XYZPoint MainVertex(0, 0, 0);
  reco::Vertex TheMainVertex = vertexHandle.product()->front();

  if (vertexHandle.isValid()) {
    const reco::VertexCollection* vertices = vertexHandle.product();
    if ((*vertices).at(0).isValid()) {
      auto theMainVtx = (*vertices).at(0);
      MainVertex.SetXYZ(theMainVtx.position().x(), theMainVtx.position().y(), theMainVtx.position().z());
    }
  }

  const math::XYZPoint myVertex(
      aTransientVertex.position().x(), aTransientVertex.position().y(), aTransientVertex.position().z());
  const math::XYZPoint deltaVtx(
      MainVertex.x() - myVertex.x(), MainVertex.y() - myVertex.y(), MainVertex.z() - myVertex.z());

  if (TheMainVertex.isValid()) {
    // Z Vertex distance in the xy plane

    VertexDistanceXY vertTool;
    double distance = vertTool.distance(aTransientVertex, TheMainVertex).value();
    double dist_err = vertTool.distance(aTransientVertex, TheMainVertex).error();

    hSVDist_->Fill(distance * cmToum);
    hSVDistSig_->Fill(distance / dist_err);

    // fill the VtxDist plots
    VtxDistPlots.fillPlots(distance * cmToum, tktk_p4);

    // fill the VtxDisSig plots
    VtxDistSigPlots.fillPlots(distance / dist_err, tktk_p4);

    // Z Vertex distance in 3D

    VertexDistance3D vertTool3D;
    double distance3D = vertTool3D.distance(aTransientVertex, TheMainVertex).value();
    double dist3D_err = vertTool3D.distance(aTransientVertex, TheMainVertex).error();

    hSVDist3D_->Fill(distance3D * cmToum);
    hSVDist3DSig_->Fill(distance3D / dist3D_err);

    // fill the VtxDist3D plots
    VtxDist3DPlots.fillPlots(distance3D * cmToum, tktk_p4);

    // fill the VtxDisSig plots
    VtxDist3DSigPlots.fillPlots(distance3D / dist3D_err, tktk_p4);

    LogDebug("DiMuonVertexValidation") << "distance: " << distance << "+/-" << dist_err << std::endl;
    // cut on the PV - SV distance
    if (distance * cmToum < maxSVdist_) {
      myCounts.eventsAfterDist++;

      double cosphi = (ZpT.x() * deltaVtx.x() + ZpT.y() * deltaVtx.y()) /
                      (sqrt(ZpT.x() * ZpT.x() + ZpT.y() * ZpT.y()) *
                       sqrt(deltaVtx.x() * deltaVtx.x() + deltaVtx.y() * deltaVtx.y()));

      double cosphi3D = (Zp.x() * deltaVtx.x() + Zp.y() * deltaVtx.y() + Zp.z() * deltaVtx.z()) /
                        (sqrt(Zp.x() * Zp.x() + Zp.y() * Zp.y() + Zp.z() * Zp.z()) *
                         sqrt(deltaVtx.x() * deltaVtx.x() + deltaVtx.y() * deltaVtx.y() + deltaVtx.z() * deltaVtx.z()));

      LogDebug("DiMuonVertexValidation") << "cos(phi) = " << cosphi << std::endl;

      hCosPhi_->Fill(cosphi);
      hCosPhi3D_->Fill(cosphi3D);

      // fill the cosphi plots
      CosPhiPlots.fillPlots(cosphi, tktk_p4);

      // fill the VtxDisSig plots
      CosPhi3DPlots.fillPlots(cosphi3D, tktk_p4);
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void DiMuonVertexValidation::beginJob() {
  edm::Service<TFileService> fs;

  // clang-format off
  TH1F::SetDefaultSumw2(kTRUE);
  hSVProb_ = fs->make<TH1F>("VtxProb", ";ZV vertex probability;N(#mu#mu pairs)", 100, 0., 1.);

  hSVDist_ = fs->make<TH1F>("VtxDist", ";PV-ZV xy distance [#mum];N(#mu#mu pairs)", 100, 0., 300.);
  hSVDistSig_ = fs->make<TH1F>("VtxDistSig", ";PV-ZV xy distance signficance;N(#mu#mu pairs)", 100, 0., 5.);

  hSVDist3D_ = fs->make<TH1F>("VtxDist3D", ";PV-ZV 3D distance [#mum];N(#mu#mu pairs)", 100, 0., 300.);
  hSVDist3DSig_ = fs->make<TH1F>("VtxDist3DSig", ";PV-ZV 3D distance signficance;N(#mu#mu pairs)", 100, 0., 5.);

  hInvMass_ = fs->make<TH1F>("InvMass", ";M(#mu#mu) [GeV];N(#mu#mu pairs)", 70., 50., 120.);
  hTrackInvMass_ = fs->make<TH1F>("TkTkInvMass", ";M(tk,tk) [GeV];N(tk tk pairs)", 70., 50., 120.);

  hCosPhi_ = fs->make<TH1F>("CosPhi", ";cos(#phi_{xy});N(#mu#mu pairs)", 50, -1., 1.);
  hCosPhi3D_ = fs->make<TH1F>("CosPhi3D", ";cos(#phi_{3D});N(#mu#mu pairs)", 50, -1., 1.);

  hCosPhiInv_ = fs->make<TH1F>("CosPhiInv", ";inverted cos(#phi_{xy});N(#mu#mu pairs)", 50, -1., 1.);
  hCosPhiInv3D_ = fs->make<TH1F>("CosPhiInv3D", ";inverted cos(#phi_{3D});N(#mu#mu pairs)", 50, -1., 1.);
  // clang-format on

  // 2D Maps

  TFileDirectory dirCosPhi = fs->mkdir("CosPhiPlots");
  CosPhiPlots.bookFromPSet(dirCosPhi, CosPhiConfiguration_);

  TFileDirectory dirCosPhi3D = fs->mkdir("CosPhi3DPlots");
  CosPhi3DPlots.bookFromPSet(dirCosPhi3D, CosPhi3DConfiguration_);

  TFileDirectory dirVtxProb = fs->mkdir("VtxProbPlots");
  VtxProbPlots.bookFromPSet(dirVtxProb, VtxProbConfiguration_);

  TFileDirectory dirVtxDist = fs->mkdir("VtxDistPlots");
  VtxDistPlots.bookFromPSet(dirVtxDist, VtxDistConfiguration_);

  TFileDirectory dirVtxDist3D = fs->mkdir("VtxDist3DPlots");
  VtxDist3DPlots.bookFromPSet(dirVtxDist3D, VtxDist3DConfiguration_);

  TFileDirectory dirVtxDistSig = fs->mkdir("VtxDistSigPlots");
  VtxDistSigPlots.bookFromPSet(dirVtxDistSig, VtxDistSigConfiguration_);

  TFileDirectory dirVtxDist3DSig = fs->mkdir("VtxDist3DSigPlots");
  VtxDist3DSigPlots.bookFromPSet(dirVtxDist3DSig, VtxDist3DSigConfiguration_);

  TFileDirectory dirInvariantMass = fs->mkdir("InvariantMassPlots");
  ZMassPlots.bookFromPSet(dirInvariantMass, DiMuMassConfiguration_);

  // cut flow

  hCutFlow_ = fs->make<TH1F>("hCutFlow", "cut flow;cut step;events left", 6, -0.5, 5.5);
  std::string names[6] = {"Total", "Mult.", ">pT", "<eta", "hasVtx", "VtxDist"};
  for (unsigned int i = 0; i < 6; i++) {
    hCutFlow_->GetXaxis()->SetBinLabel(i + 1, names[i].c_str());
  }

  myCounts.zeroAll();
}

// ------------ method called once each job just after ending the event loop  ------------
void DiMuonVertexValidation::endJob() {
  myCounts.printCounts();

  hCutFlow_->SetBinContent(1, myCounts.eventsTotal);
  hCutFlow_->SetBinContent(2, myCounts.eventsAfterMult);
  hCutFlow_->SetBinContent(3, myCounts.eventsAfterPt);
  hCutFlow_->SetBinContent(4, myCounts.eventsAfterEta);
  hCutFlow_->SetBinContent(5, myCounts.eventsAfterVtx);
  hCutFlow_->SetBinContent(6, myCounts.eventsAfterDist);

  TH1F::SetDefaultSumw2(kTRUE);
  const unsigned int nBinsX = hCosPhi_->GetNbinsX();
  for (unsigned int i = 1; i <= nBinsX; i++) {
    //float binContent = hCosPhi_->GetBinContent(i);
    float invertedBinContent = hCosPhi_->GetBinContent(nBinsX + 1 - i);
    float invertedBinError = hCosPhi_->GetBinError(nBinsX + 1 - i);
    hCosPhiInv_->SetBinContent(i, invertedBinContent);
    hCosPhiInv_->SetBinError(i, invertedBinError);

    //float binContent3D = hCosPhi3D_->GetBinContent(i);
    float invertedBinContent3D = hCosPhi3D_->GetBinContent(nBinsX + 1 - i);
    float invertedBinError3D = hCosPhi3D_->GetBinError(nBinsX + 1 - i);
    hCosPhiInv3D_->SetBinContent(i, invertedBinContent3D);
    hCosPhiInv3D_->SetBinError(i, invertedBinError3D);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DiMuonVertexValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.ifValue(edm::ParameterDescription<bool>("useReco", true, true),
               true >> edm::ParameterDescription<edm::InputTag>("muons", edm::InputTag("muons"), true) or
                   false >> edm::ParameterDescription<edm::InputTag>(
                                "muonTracks", edm::InputTag("ALCARECOTkAlDiMuon"), true))
      ->setComment("If useReco is true need to specify the muon tracks, otherwise take the ALCARECO Inner tracks");
  //desc.add<bool>("useReco",true);
  //desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  //desc.add<edm::InputTag>("muonTracks", edm::InputTag("ALCARECOTkAlDiMuon"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<std::vector<double>>("pTThresholds", {30., 10.});
  desc.add<double>("maxSVdist", 50.);

  {
    edm::ParameterSetDescription psDiMuMass;
    psDiMuMass.add<std::string>("name", "DiMuMass");
    psDiMuMass.add<std::string>("title", "M(#mu#mu)");
    psDiMuMass.add<std::string>("yUnits", "[GeV]");
    psDiMuMass.add<int>("NxBins", 24);
    psDiMuMass.add<int>("NyBins", 50);
    psDiMuMass.add<double>("ymin", 70.);
    psDiMuMass.add<double>("ymax", 120.);
    desc.add<edm::ParameterSetDescription>("DiMuMassConfig", psDiMuMass);
  }
  {
    edm::ParameterSetDescription psCosPhi;
    psCosPhi.add<std::string>("name", "CosPhi");
    psCosPhi.add<std::string>("title", "cos(#phi_{xy})");
    psCosPhi.add<std::string>("yUnits", "");
    psCosPhi.add<int>("NxBins", 50);
    psCosPhi.add<int>("NyBins", 50);
    psCosPhi.add<double>("ymin", -1.);
    psCosPhi.add<double>("ymax", 1.);
    desc.add<edm::ParameterSetDescription>("CosPhiConfig", psCosPhi);
  }
  {
    edm::ParameterSetDescription psCosPhi3D;
    psCosPhi3D.add<std::string>("name", "CosPhi3D");
    psCosPhi3D.add<std::string>("title", "cos(#phi_{3D})");
    psCosPhi3D.add<std::string>("yUnits", "");
    psCosPhi3D.add<int>("NxBins", 50);
    psCosPhi3D.add<int>("NyBins", 50);
    psCosPhi3D.add<double>("ymin", -1.);
    psCosPhi3D.add<double>("ymax", 1.);
    desc.add<edm::ParameterSetDescription>("CosPhi3DConfig", psCosPhi3D);
  }
  {
    edm::ParameterSetDescription psVtxProb;
    psVtxProb.add<std::string>("name", "VtxProb");
    psVtxProb.add<std::string>("title", "Prob(#chi^{2}_{SV})");
    psVtxProb.add<std::string>("yUnits", "");
    psVtxProb.add<int>("NxBins", 50);
    psVtxProb.add<int>("NyBins", 50);
    psVtxProb.add<double>("ymin", 0);
    psVtxProb.add<double>("ymax", 1.);
    desc.add<edm::ParameterSetDescription>("VtxProbConfig", psVtxProb);
  }
  {
    edm::ParameterSetDescription psVtxDist;
    psVtxDist.add<std::string>("name", "VtxDist");
    psVtxDist.add<std::string>("title", "d_{xy}(PV,SV)");
    psVtxDist.add<std::string>("yUnits", "[#mum]");
    psVtxDist.add<int>("NxBins", 50);
    psVtxDist.add<int>("NyBins", 100);
    psVtxDist.add<double>("ymin", 0);
    psVtxDist.add<double>("ymax", 300.);
    desc.add<edm::ParameterSetDescription>("VtxDistConfig", psVtxDist);
  }
  {
    edm::ParameterSetDescription psVtxDist3D;
    psVtxDist3D.add<std::string>("name", "VtxDist3D");
    psVtxDist3D.add<std::string>("title", "d_{3D}(PV,SV)");
    psVtxDist3D.add<std::string>("yUnits", "[#mum]");
    psVtxDist3D.add<int>("NxBins", 50);
    psVtxDist3D.add<int>("NyBins", 250);
    psVtxDist3D.add<double>("ymin", 0);
    psVtxDist3D.add<double>("ymax", 500.);
    desc.add<edm::ParameterSetDescription>("VtxDist3DConfig", psVtxDist3D);
  }
  {
    edm::ParameterSetDescription psVtxDistSig;
    psVtxDistSig.add<std::string>("name", "VtxDistSig");
    psVtxDistSig.add<std::string>("title", "d_{xy}(PV,SV)/#sigma_{dxy}(PV,SV)");
    psVtxDistSig.add<std::string>("yUnits", "");
    psVtxDistSig.add<int>("NxBins", 50);
    psVtxDistSig.add<int>("NyBins", 100);
    psVtxDistSig.add<double>("ymin", 0);
    psVtxDistSig.add<double>("ymax", 5.);
    desc.add<edm::ParameterSetDescription>("VtxDistSigConfig", psVtxDistSig);
  }
  {
    edm::ParameterSetDescription psVtxDist3DSig;
    psVtxDist3DSig.add<std::string>("name", "VtxDist3DSig");
    psVtxDist3DSig.add<std::string>("title", "d_{3D}(PV,SV)/#sigma_{d3D}(PV,SV)");
    psVtxDist3DSig.add<std::string>("yUnits", "");
    psVtxDist3DSig.add<int>("NxBins", 50);
    psVtxDist3DSig.add<int>("NyBins", 100);
    psVtxDist3DSig.add<double>("ymin", 0);
    psVtxDist3DSig.add<double>("ymax", 5.);
    desc.add<edm::ParameterSetDescription>("VtxDist3DSigConfig", psVtxDist3DSig);
  }

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DiMuonVertexValidation);
