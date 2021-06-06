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
#include <fmt/printf.h>

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
// Ancillary class for plotting
//
class PlotsVsDiMuKinematics {
public:
  PlotsVsDiMuKinematics() : m_name(""), m_title(""), m_ytitle(""), m_isBooked(false) {}

  //________________________________________________________________________________//
  // overloaded constructor
  PlotsVsDiMuKinematics(const std::string& name, const std::string& tt, const std::string& ytt)
      : m_name(name), m_title(tt), m_ytitle(ytt), m_isBooked(false) {}

  ~PlotsVsDiMuKinematics() = default;

  //________________________________________________________________________________//
  void bookFromPSet(const TFileDirectory& fs, const edm::ParameterSet& hpar) {
    std::string namePostfix;
    std::string titlePostfix;
    float xmin, xmax;

    for (const auto& xAx : axisChoices) {
      switch (xAx) {
        case xAxis::Z_PHI:
          xmin = -M_PI;
          xmax = M_PI;
          namePostfix = "MuMuPhi";
          titlePostfix = "#mu#mu pair #phi;#mu^{+}#mu^{-} #phi";
          break;
        case xAxis::Z_ETA:
          xmin = -3.5;
          xmax = 3.5;
          namePostfix = "MuMuEta";
          titlePostfix = "#mu#mu pair #eta;#mu^{+}#mu^{-} #eta";
          break;
        case xAxis::MP_PHI:
          xmin = -M_PI;
          xmax = M_PI;
          namePostfix = "MuPlusPhi";
          titlePostfix = "#mu^{+} #phi;#mu^{+} #phi [rad]";
          break;
        case xAxis::MP_ETA:
          xmin = -2.4;
          xmax = 2.4;
          namePostfix = "MuPlusEta";
          titlePostfix = "#mu^{+} #eta;#mu^{+} #eta";
          break;
        case xAxis::MM_PHI:
          xmin = -M_PI;
          xmax = M_PI;
          namePostfix = "MuMinusPhi";
          titlePostfix = "#mu^{-} #phi;#mu^{-} #phi [rad]";
          break;
        case xAxis::MM_ETA:
          xmin = -2.4;
          xmax = 2.4;
          namePostfix = "MuMinusEta";
          titlePostfix = "#mu^{-} #eta;#mu^{+} #eta";
          break;
        default:
          throw cms::Exception("LogicalError") << " there is not such Axis choice as " << xAx;
      }

      const auto& h2name = fmt::sprintf("%sVs%s", hpar.getParameter<std::string>("name"), namePostfix);
      const auto& h2title = fmt::sprintf("%s vs %s;%s% s",
                                         hpar.getParameter<std::string>("title"),
                                         titlePostfix,
                                         hpar.getParameter<std::string>("title"),
                                         hpar.getParameter<std::string>("yUnits"));

      m_h2_map[xAx] = fs.make<TH2F>(h2name.c_str(),
                                    h2title.c_str(),
                                    hpar.getParameter<int32_t>("NxBins"),
                                    xmin,
                                    xmax,
                                    hpar.getParameter<int32_t>("NyBins"),
                                    hpar.getParameter<double>("ymin"),
                                    hpar.getParameter<double>("ymax"));
    }

    // flip the is booked bit
    m_isBooked = true;
  }

  //________________________________________________________________________________//
  void bookPlots(TFileDirectory& fs, const float valmin, const float valmax, const int nxbins, const int nybins) {
    if (m_name.empty() && m_title.empty() && m_ytitle.empty()) {
      edm::LogError("PlotsVsDiMuKinematics")
          << "In" << __FUNCTION__ << "," << __LINE__
          << "trying to book plots without the right constructor being called!" << std::endl;
      return;
    }

    static constexpr float maxMuEta = 2.4;
    static constexpr float maxMuMuEta = 3.5;
    TH1F::SetDefaultSumw2(kTRUE);

    // clang-format off
    m_h2_map[xAxis::Z_ETA] = fs.make<TH2F>(fmt::sprintf("%sVsMuMuEta", m_name).c_str(),
					   fmt::sprintf("%s vs #mu#mu pair #eta;#mu^{+}#mu^{-} #eta;%s", m_title, m_ytitle).c_str(),
					   nxbins, -M_PI, M_PI,
					   nybins, valmin, valmax);

    m_h2_map[xAxis::Z_PHI] = fs.make<TH2F>(fmt::sprintf("%sVsMuMuPhi", m_name).c_str(),
					   fmt::sprintf("%s vs #mu#mu pair #phi;#mu^{+}#mu^{-} #phi [rad];%s", m_title, m_ytitle).c_str(),
					   nxbins, -maxMuMuEta, maxMuMuEta,
					   nybins, valmin, valmax);

    m_h2_map[xAxis::MP_ETA] = fs.make<TH2F>(fmt::sprintf("%sVsMuPlusEta", m_name).c_str(),
					    fmt::sprintf("%s vs #mu^{+} #eta;#mu^{+} #eta;%s", m_title, m_ytitle).c_str(),
					    nxbins, -maxMuEta, maxMuEta,
					    nybins, valmin, valmax);

    m_h2_map[xAxis::MP_PHI] = fs.make<TH2F>(fmt::sprintf("%sVsMuPlusPhi", m_name).c_str(),
					    fmt::sprintf("%s vs #mu^{+} #phi;#mu^{+} #phi [rad];%s", m_title, m_ytitle).c_str(),
					    nxbins, -M_PI, M_PI,
					    nybins, valmin, valmax);

    m_h2_map[xAxis::MM_ETA] = fs.make<TH2F>(fmt::sprintf("%sVsMuMinusEta", m_name).c_str(),
					    fmt::sprintf("%s vs #mu^{-} #eta;#mu^{-} #eta;%s", m_title, m_ytitle).c_str(),
					    nxbins, -maxMuEta, maxMuEta,
					    nybins, valmin, valmax);

    m_h2_map[xAxis::MM_PHI] = fs.make<TH2F>(fmt::sprintf("%sVsMuMinusPhi", m_name).c_str(),
					    fmt::sprintf("%s vs #mu^{-} #phi;#mu^{-} #phi [rad];%s", m_title, m_ytitle).c_str(),
					    nxbins, -M_PI, M_PI,
					    nybins, valmin, valmax);
    // clang-format on

    // flip the is booked bit
    m_isBooked = true;
  }

  //________________________________________________________________________________//
  void fillPlots(const float val, const std::pair<TLorentzVector, TLorentzVector>& momenta) {
    if (!m_isBooked) {
      edm::LogError("PlotsVsDiMuKinematics")
          << "In" << __FUNCTION__ << "," << __LINE__ << "trying to fill a plot not booked!" << std::endl;
      return;
    }

    m_h2_map[xAxis::Z_ETA]->Fill((momenta.first + momenta.second).Eta(), val);
    m_h2_map[xAxis::Z_PHI]->Fill((momenta.first + momenta.second).Phi(), val);
    m_h2_map[xAxis::MP_ETA]->Fill((momenta.first).Eta(), val);
    m_h2_map[xAxis::MP_PHI]->Fill((momenta.first).Phi(), val);
    m_h2_map[xAxis::MM_ETA]->Fill((momenta.second).Eta(), val);
    m_h2_map[xAxis::MM_PHI]->Fill((momenta.second).Phi(), val);
  }

private:
  enum xAxis { Z_PHI, Z_ETA, MP_PHI, MP_ETA, MM_PHI, MM_ETA };
  const std::vector<xAxis> axisChoices = {
      xAxis::Z_PHI, xAxis::Z_ETA, xAxis::MP_PHI, xAxis::MP_ETA, xAxis::MM_PHI, xAxis::MM_ETA};

  const std::string m_name;
  const std::string m_title;
  const std::string m_ytitle;

  bool m_isBooked;

  std::map<xAxis, TH2F*> m_h2_map;
};

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

  // 2D maps

  PlotsVsDiMuKinematics CosPhiPlots = PlotsVsDiMuKinematics();
  PlotsVsDiMuKinematics CosPhi3DPlots = PlotsVsDiMuKinematics();
  PlotsVsDiMuKinematics VtxProbPlots = PlotsVsDiMuKinematics();
  PlotsVsDiMuKinematics VtxDistPlots = PlotsVsDiMuKinematics();
  PlotsVsDiMuKinematics VtxDist3DPlots = PlotsVsDiMuKinematics();
  PlotsVsDiMuKinematics VtxDistSigPlots = PlotsVsDiMuKinematics();
  PlotsVsDiMuKinematics VtxDist3DSigPlots = PlotsVsDiMuKinematics();
  PlotsVsDiMuKinematics ZMassPlots = PlotsVsDiMuKinematics();

  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbESToken_;

  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;   //used to select what tracks to read from configuration file
  edm::EDGetTokenT<reco::MuonCollection> muonsToken_;     //used to select what tracks to read from configuration file
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;  //used to select what vertices to read from configuration file
};

//
// constants, enums and typedefs
//

static constexpr float cmToum = 10e4;

//
// static data member definitions
//

//
// constructors and destructor
//
DiMuonVertexValidation::DiMuonVertexValidation(const edm::ParameterSet& iConfig)
    : pTthresholds_(iConfig.getParameter<std::vector<double>>("pTThresholds")),
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
      muonsToken_(consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      vertexToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))) {
  usesResource(TFileService::kSharedResource);

  // sort the vector of thresholds
  std::sort(pTthresholds_.begin(), pTthresholds_.end(), [](const double& lhs, const double& rhs) { return lhs > rhs; });

  edm::LogInfo("DiMuonVertexValidation") << __FUNCTION__;
  for (const auto& thr : pTthresholds_) {
    edm::LogInfo("DiMuonVertexValidation") << " Threshold: " << thr << " ";
  }
  edm::LogInfo("DiMuonVertexValidation") << "\n Max SV distance: " << maxSVdist_ << " ";
}

DiMuonVertexValidation::~DiMuonVertexValidation() = default;

//
// member functions
//

// ------------ method called for each event  ------------
void DiMuonVertexValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

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
  if ((myGoodMuonVector[0]->pt()) < pTthresholds_[0] || (myGoodMuonVector[1]->pt() < pTthresholds_[1]))
    return;
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
  std::vector<const reco::Track*> myTracks;

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

  static constexpr float mumass = 0.105658367;  //mu mass  (GeV/c^2)

  TLorentzVector p4_tplus(tplus->px(), tplus->py(), tplus->pz(), sqrt((tplus->p() * tplus->p()) + (mumass * mumass)));
  TLorentzVector p4_tminus(
      tminus->px(), tminus->py(), tminus->pz(), sqrt((tminus->p() * tminus->p()) + (mumass * mumass)));

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
}

// ------------ method called once each job just after ending the event loop  ------------
void DiMuonVertexValidation::endJob() {
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
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
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
