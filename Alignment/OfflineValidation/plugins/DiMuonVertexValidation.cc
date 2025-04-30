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

// IP tools
#include "TrackingTools/IPTools/interface/IPTools.h"

// ROOT
#include "TH1F.h"
#include "TH2F.h"

//#define LogDebug(X) std::cout << X

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
  const reco::Vertex* findClosestVertex(const TransientVertex aTransVtx, const reco::VertexCollection* vertices) const;
  void endJob() override;

  // ----------member data ---------------------------
  DiLeptonHelp::Counts myCounts;

  const std::string motherName_;
  const bool useReco_;
  const bool useClosestVertex_;
  std::pair<float, float> massLimits_; /* for the mass plot x-range */
  std::vector<double> pTthresholds_;
  const float maxSVdist_;

  // plot configurations
  const edm::ParameterSet CosPhiConfiguration_;
  const edm::ParameterSet CosPhi3DConfiguration_;
  const edm::ParameterSet VtxProbConfiguration_;
  const edm::ParameterSet VtxDistConfiguration_;
  const edm::ParameterSet VtxDist3DConfiguration_;
  const edm::ParameterSet VtxDistSigConfiguration_;
  const edm::ParameterSet VtxDist3DSigConfiguration_;
  const edm::ParameterSet DiMuMassConfiguration_;

  // control plots
  TH1F* hSVProb_;
  TH1F* hSVChi2_;
  TH1F* hSVNormChi2_;

  TH1F* hSVDist_;
  TH1F* hSVDistErr_;
  TH1F* hSVDistSig_;
  TH1F* hSVCompatibility_;

  TH1F* hSVDist3D_;
  TH1F* hSVDist3DErr_;
  TH1F* hSVDist3DSig_;
  TH1F* hSVCompatibility3D_;

  TH1F* hCosPhi_;
  TH1F* hCosPhi3D_;
  TH1F* hCosPhiInv_;
  TH1F* hCosPhiInv3D_;
  TH1F* hCosPhiUnbalance_;
  TH1F* hCosPhi3DUnbalance_;

  TH1F* hInvMass_;
  TH1F* hTrackInvMass_;

  TH1F* hCutFlow_;

  // impact parameters information
  TH1F* hdxy_;
  TH1F* hdz_;
  TH1F* hdxyErr_;
  TH1F* hdzErr_;
  TH1F* hIP2d_;
  TH1F* hIP3d_;
  TH1F* hIP2dsig_;
  TH1F* hIP3dsig_;

  // 2D maps

  DiLeptonHelp::PlotsVsKinematics CosPhiPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics CosPhi3DPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics VtxProbPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics VtxDistPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics VtxDist3DPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics VtxDistSigPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics VtxDist3DSigPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);
  DiLeptonHelp::PlotsVsKinematics ZMassPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::MM);

  // plots vs region
  DiLeptonHelp::PlotsVsDiLeptonRegion CosPhi3DInEtaBins = DiLeptonHelp::PlotsVsDiLeptonRegion(1.5);
  DiLeptonHelp::PlotsVsDiLeptonRegion InvMassInEtaBins = DiLeptonHelp::PlotsVsDiLeptonRegion(1.5);

  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbESToken_;

  //used to select what tracks to read from configuration file
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  //used to select what vertices to read from configuration file
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  // either on or the other!
  edm::EDGetTokenT<reco::MuonCollection> muonsToken_;      // used to select tracks to read from configuration file
  edm::EDGetTokenT<reco::TrackCollection> alcaRecoToken_;  //used to select muon tracks to read from configuration file
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
    : motherName_(iConfig.getParameter<std::string>("decayMotherName")),
      useReco_(iConfig.getParameter<bool>("useReco")),
      useClosestVertex_(iConfig.getParameter<bool>("useClosestVertex")),
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
      vertexToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))) {
  if (useReco_) {
    tracksToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));
    muonsToken_ = consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"));
  } else {
    alcaRecoToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("muonTracks"));
  }

  usesResource(TFileService::kSharedResource);

  // sort the vector of thresholds
  std::sort(pTthresholds_.begin(), pTthresholds_.end(), [](const double& lhs, const double& rhs) { return lhs > rhs; });

  edm::LogInfo("DiMuonVertexValidation") << __FUNCTION__;
  for (const auto& thr : pTthresholds_) {
    edm::LogInfo("DiMuonVertexValidation") << " Threshold: " << thr << " ";
  }
  edm::LogInfo("DiMuonVertexValidation") << "Max SV distance: " << maxSVdist_ << " ";

  // set the limits for the mass plots
  if (motherName_.find('Z') != std::string::npos) {
    massLimits_ = std::make_pair(50., 120);
  } else if (motherName_.find("J/#psi") != std::string::npos) {
    massLimits_ = std::make_pair(2.7, 3.4);
  } else if (motherName_.find("#Upsilon") != std::string::npos) {
    massLimits_ = std::make_pair(8.9, 9.9);
  } else {
    edm::LogError("DiMuonVertexValidation") << " unrecognized decay mother particle: " << motherName_
                                            << " setting the default for the Z->mm (50.,120.)" << std::endl;
    massLimits_ = std::make_pair(50., 120);
  }
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

#ifdef EDM_ML_DEBUG
  for (const auto& track : myTracks) {
    edm::LogVerbatim("DiMuonVertexValidation") << __PRETTY_FUNCTION__ << " pT: " << track->pt() << " GeV"
                                               << " , pT error: " << track->ptError() << " GeV"
                                               << " , eta: " << track->eta() << " , phi: " << track->phi() << std::endl;
  }
#endif

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

#ifdef EDM_ML_DEBUG
  // Define a lambda function to convert TLorentzVector to a string
  auto tLorentzVectorToString = [](const TLorentzVector& vector) {
    return std::to_string(vector.Px()) + " " + std::to_string(vector.Py()) + " " + std::to_string(vector.Pz()) + " " +
           std::to_string(vector.E());
  };

  edm::LogVerbatim("DiMuonVertexValidation") << "mu+" << tLorentzVectorToString(p4_tplus) << std::endl;
  edm::LogVerbatim("DiMuonVertexValidation") << "mu-" << tLorentzVectorToString(p4_tminus) << std::endl;
#endif

  const auto& Zp4 = p4_tplus + p4_tminus;
  float track_invMass = Zp4.M();
  hTrackInvMass_->Fill(track_invMass);

  // creat the pair of TLorentVectors used to make the plos
  std::pair<TLorentzVector, TLorentzVector> tktk_p4 = std::make_pair(p4_tplus, p4_tminus);

  // fill the z->mm mass plots
  ZMassPlots.fillPlots(track_invMass, tktk_p4);
  InvMassInEtaBins.fillTH1Plots(track_invMass, tktk_p4);

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
  hSVChi2_->Fill(aTransientVertex.totalChiSquared());
  hSVNormChi2_->Fill(aTransientVertex.totalChiSquared() / (int)aTransientVertex.degreesOfFreedom());

  LogDebug("DiMuonVertexValidation") << " vertex norm chi2: "
                                     << (aTransientVertex.totalChiSquared() / (int)aTransientVertex.degreesOfFreedom())
                                     << std::endl;

  if (!aTransientVertex.isValid())
    return;

  myCounts.eventsAfterVtx++;

  // fill the VtxProb plots
  VtxProbPlots.fillPlots(SVProb, tktk_p4);

  math::XYZPoint mainVtxPos(0, 0, 0);
  const reco::Vertex* theClosestVertex = nullptr;
  // get collection of reconstructed vertices from event
  edm::Handle<reco::VertexCollection> vertexHandle = iEvent.getHandle(vertexToken_);
  if (vertexHandle.isValid()) {
    const reco::VertexCollection* vertices = vertexHandle.product();
    theClosestVertex = this->findClosestVertex(aTransientVertex, vertices);
  } else {
    edm::LogWarning("DiMuonVertexMonitor") << "invalid vertex collection encountered Skipping event!";
    return;
  }

  reco::Vertex theMainVertex;
  if (!useClosestVertex_ || theClosestVertex == nullptr) {
    // if the closest vertex is not available, or explicitly not chosen
    theMainVertex = vertexHandle.product()->front();
  } else {
    theMainVertex = *theClosestVertex;
  }

  mainVtxPos.SetXYZ(theMainVertex.position().x(), theMainVertex.position().y(), theMainVertex.position().z());
  const math::XYZPoint myVertex(
      aTransientVertex.position().x(), aTransientVertex.position().y(), aTransientVertex.position().z());
  const math::XYZPoint deltaVtx(
      mainVtxPos.x() - myVertex.x(), mainVtxPos.y() - myVertex.y(), mainVtxPos.z() - myVertex.z());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DiMuonVertexValidation")
      << "mm vertex position:" << aTransientVertex.position().x() << "," << aTransientVertex.position().y() << ","
      << aTransientVertex.position().z();

  edm::LogVerbatim("DiMuonVertexValidation") << "main vertex position:" << theMainVertex.position().x() << ","
                                             << theMainVertex.position().y() << "," << theMainVertex.position().z();
#endif

  if (theMainVertex.isValid()) {
    // fill the impact parameter plots
    for (const auto& track : myTracks) {
      hdxy_->Fill(track->dxy(mainVtxPos) * cmToum);
      hdz_->Fill(track->dz(mainVtxPos) * cmToum);
      hdxyErr_->Fill(track->dxyError() * cmToum);
      hdzErr_->Fill(track->dzError() * cmToum);

      const auto& ttrk = theB->build(track);
      Global3DVector dir(track->px(), track->py(), track->pz());
      const auto& ip2d = IPTools::signedTransverseImpactParameter(ttrk, dir, theMainVertex);
      const auto& ip3d = IPTools::signedImpactParameter3D(ttrk, dir, theMainVertex);

      hIP2d_->Fill(ip2d.second.value() * cmToum);
      hIP3d_->Fill(ip3d.second.value() * cmToum);
      hIP2dsig_->Fill(ip2d.second.significance());
      hIP3dsig_->Fill(ip3d.second.significance());
    }

    LogDebug("DiMuonVertexValidation") << " after filling the IP histograms " << std::endl;

    // Z Vertex distance in the xy plane
    VertexDistanceXY vertTool;
    double distance = vertTool.distance(aTransientVertex, theMainVertex).value();
    double dist_err = vertTool.distance(aTransientVertex, theMainVertex).error();
    float compatibility = 0.;

    try {
      compatibility = vertTool.compatibility(aTransientVertex, theMainVertex);
    } catch (cms::Exception& er) {
      LogTrace("DiMuonVertexValidation") << "caught std::exception " << er.what() << std::endl;
    }

    hSVCompatibility_->Fill(compatibility);
    hSVDist_->Fill(distance * cmToum);
    hSVDistErr_->Fill(dist_err * cmToum);
    hSVDistSig_->Fill(distance / dist_err);

    // fill the VtxDist plots
    VtxDistPlots.fillPlots(distance * cmToum, tktk_p4);

    // fill the VtxDisSig plots
    VtxDistSigPlots.fillPlots(distance / dist_err, tktk_p4);

    // Z Vertex distance in 3D

    VertexDistance3D vertTool3D;
    double distance3D = vertTool3D.distance(aTransientVertex, theMainVertex).value();
    double dist3D_err = vertTool3D.distance(aTransientVertex, theMainVertex).error();
    float compatibility3D = 0.;

    try {
      compatibility3D = vertTool3D.compatibility(aTransientVertex, theMainVertex);
    } catch (cms::Exception& er) {
      LogTrace("DiMuonVertexMonitor") << "caught std::exception " << er.what() << std::endl;
    }

    hSVCompatibility3D_->Fill(compatibility3D);
    hSVDist3D_->Fill(distance3D * cmToum);
    hSVDist3DErr_->Fill(dist3D_err * cmToum);
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

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DiMuonVertexValidation")
          << "distance " << distance * cmToum << " cosphi3D:" << cosphi3D << std::endl;
#endif

      // unbalance
      hCosPhiUnbalance_->Fill(cosphi, 1.);
      hCosPhiUnbalance_->Fill(-cosphi, -1.);
      hCosPhi3DUnbalance_->Fill(cosphi3D, 1.);
      hCosPhi3DUnbalance_->Fill(-cosphi3D, -1.);

      // fill the cosphi plots
      CosPhiPlots.fillPlots(cosphi, tktk_p4);

      // fill the cosphi3D plots
      CosPhi3DPlots.fillPlots(cosphi3D, tktk_p4);

      // fill the cosphi3D plots in eta bins
      CosPhi3DInEtaBins.fillTH1Plots(cosphi3D, tktk_p4);
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void DiMuonVertexValidation::beginJob() {
  edm::Service<TFileService> fs;

  // clang-format off
  TH1F::SetDefaultSumw2(kTRUE);
  hSVProb_ = fs->make<TH1F>("VtxProb", ";#mu^{+}#mu^{-} vertex probability;N(#mu#mu pairs)", 100, 0., 1.);

  auto extractRangeValues = [](const edm::ParameterSet& PSetConfiguration_) -> std::pair<double, double> {
    double min = PSetConfiguration_.getParameter<double>("ymin");
    double max = PSetConfiguration_.getParameter<double>("ymax");
    return {min, max};
  };

  std::string ts = fmt::sprintf(";%s vertex probability;N(#mu#mu pairs)", motherName_);
  std::string ps = "N(#mu#mu pairs)";
  ts = fmt::sprintf("#chi^{2} of the %s vertex; #chi^{2} of the %s vertex; %s", motherName_, motherName_, ps);
  hSVChi2_ = fs->make<TH1F>("VtxChi2", ts.c_str(), 200, 0., 200.);

  ts = fmt::sprintf("#chi^{2}/ndf of the %s vertex; #chi^{2}/ndf of %s vertex; %s", motherName_, motherName_, ps);
  hSVNormChi2_ = fs->make<TH1F>("VtxNormChi2", ts.c_str(), 100, 0., 20.);

  // take the range from the 2D histograms
  const auto& svDistRng = extractRangeValues(VtxDistConfiguration_);
  hSVDist_ = fs->make<TH1F>("VtxDist", ";PV-#mu^{+}#mu^{-} vertex xy distance [#mum];N(#mu#mu pairs)", 100, svDistRng.first, svDistRng.second);

  std::string histTit = motherName_ + " #rightarrow #mu^{+}#mu^{-}";
  ts = fmt::sprintf("%s;PV-%sV xy distance error [#mum];%s", histTit, motherName_, ps);
  hSVDistErr_ = fs->make<TH1F>("VtxDistErr", ts.c_str(), 100, 0., 1000.);

  // take the range from the 2D histograms
  const auto& svDistSigRng = extractRangeValues(VtxDistSigConfiguration_);
  hSVDistSig_ = fs->make<TH1F>("VtxDistSig", ";PV-#mu^{+}#mu^{-} vertex xy distance signficance;N(#mu#mu pairs)", 100, svDistSigRng.first, svDistSigRng.second);

  // take the range from the 2D histograms
  const auto& svDist3DRng = extractRangeValues(VtxDist3DConfiguration_);
  hSVDist3D_ = fs->make<TH1F>("VtxDist3D", ";PV-#mu^{+}#mu^{-} vertex 3D distance [#mum];N(#mu#mu pairs)", 100, svDist3DRng.first, svDist3DRng.second);

  ts = fmt::sprintf("%s;PV-%sV 3D distance error [#mum];%s", histTit, motherName_, ps);
  hSVDist3DErr_ = fs->make<TH1F>("VtxDist3DErr", ts.c_str(), 100, 0., 1000.);

  // take the range from the 2D histograms
  const auto& svDist3DSigRng = extractRangeValues(VtxDist3DSigConfiguration_);
  hSVDist3DSig_ = fs->make<TH1F>("VtxDist3DSig", ";PV-#mu^{+}#mu^{-} vertex 3D distance signficance;N(#mu#mu pairs)", 100, svDist3DSigRng.first, svDist3DSigRng.second);

  ts = fmt::sprintf("compatibility of %s vertex; compatibility of %s vertex; %s", motherName_, motherName_, ps);
  hSVCompatibility_ = fs->make<TH1F>("VtxCompatibility", ts.c_str(), 100, 0., 100.);

  ts = fmt::sprintf("3D compatibility of %s vertex;3D compatibility of %s vertex; %s", motherName_, motherName_, ps);
  hSVCompatibility3D_ = fs->make<TH1F>("VtxCompatibility3D", ts.c_str(), 100, 0., 100.);

  // take the range from the 2D histograms
  const auto& massRng = extractRangeValues(DiMuMassConfiguration_);
  hInvMass_ = fs->make<TH1F>("InvMass", ";M(#mu#mu) [GeV];N(#mu#mu pairs)", 70., massRng.first, massRng.second);
  hTrackInvMass_ = fs->make<TH1F>("TkTkInvMass", ";M(tk,tk) [GeV];N(tk tk pairs)", 70., massRng.first, massRng.second);

  hCosPhi_ = fs->make<TH1F>("CosPhi", ";cos(#phi_{xy});N(#mu#mu pairs)", 50, -1., 1.);
  hCosPhi3D_ = fs->make<TH1F>("CosPhi3D", ";cos(#phi_{3D});N(#mu#mu pairs)", 50, -1., 1.);

  hCosPhiInv_ = fs->make<TH1F>("CosPhiInv", ";inverted cos(#phi_{xy});N(#mu#mu pairs)", 50, -1., 1.);
  hCosPhiInv3D_ = fs->make<TH1F>("CosPhiInv3D", ";inverted cos(#phi_{3D});N(#mu#mu pairs)", 50, -1., 1.);

  hCosPhiUnbalance_ = fs->make<TH1F>("CosPhiUnbalance", fmt::sprintf("%s;cos(#phi_{xy}) unbalance;#Delta%s", histTit, ps).c_str(), 50, -1.,1.);
  hCosPhi3DUnbalance_ = fs->make<TH1F>("CosPhi3DUnbalance", fmt::sprintf("%s;cos(#phi_{3D}) unbalance;#Delta%s", histTit, ps).c_str(), 50, -1., 1.);

  hdxy_ = fs->make<TH1F>("dxy", fmt::sprintf("%s;muon track d_{xy}(PV) [#mum];muon tracks", histTit).c_str(), 150, -300, 300);
  hdz_ = fs->make<TH1F>("dz", fmt::sprintf("%s;muon track d_{z}(PV) [#mum];muon tracks", histTit).c_str(), 150, -300, 300);
  hdxyErr_ = fs->make<TH1F>("dxyErr", fmt::sprintf("%s;muon track err_{dxy} [#mum];muon tracks", histTit).c_str(), 250, 0., 500.);
  hdzErr_ = fs->make<TH1F>("dzErr", fmt::sprintf("%s;muon track err_{dz} [#mum];muon tracks", histTit).c_str(), 250, 0., 500.);
  hIP2d_ = fs->make<TH1F>("IP2d", fmt::sprintf("%s;muon track IP_{2D} [#mum];muon tracks", histTit).c_str(), 150, -300, 300);
  hIP3d_ = fs->make<TH1F>("IP3d", fmt::sprintf("%s;muon track IP_{3D} [#mum];muon tracks", histTit).c_str(), 150, -300, 300);
  hIP2dsig_ = fs->make<TH1F>("IP2Dsig", fmt::sprintf("%s;muon track IP_{2D} significance;muon tracks", histTit).c_str(), 100, 0., 5.);
  hIP3dsig_ = fs->make<TH1F>("IP3Dsig", fmt::sprintf("%s;muon track IP_{3D} significance;muon tracks", histTit).c_str(), 100, 0., 5.);
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

  // CosPhi3D in eta bins
  TFileDirectory dirCosphi3DEta = fs->mkdir("CosPhi3DInEtaBins");
  CosPhi3DInEtaBins.bookSet(dirCosphi3DEta, hCosPhi3D_);

  // Z-> mm mass in eta bins
  TFileDirectory dirResMassEta = fs->mkdir("TkTkMassInEtaBins");
  InvMassInEtaBins.bookSet(dirResMassEta, hTrackInvMass_);

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

// compute the closest vertex to di-lepton ------------------------------------
const reco::Vertex* DiMuonVertexValidation::findClosestVertex(const TransientVertex aTransVtx,
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
  desc.add<std::string>("decayMotherName", "Z");
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<std::vector<double>>("pTThresholds", {30., 10.});
  desc.add<bool>("useClosestVertex", true);
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
-- dummy change --
