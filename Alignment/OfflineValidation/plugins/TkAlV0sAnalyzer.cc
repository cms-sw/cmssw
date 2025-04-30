// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      TkAlV0sAnalyzer
//
/*
 *\class TkAlV0sAnalyzer TkAlV0sAnalyzer.cc Alignment/TkAlV0sAnalyzer/plugins/TkAlV0sAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Thu, 14 Dec 2023 15:10:34 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/V0Candidate/interface/V0Candidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TLorentzVector.h"

//
// class declaration
//

using reco::TrackCollection;

struct MEbinning {
  int nbins;
  double xmin;
  double xmax;
};

class TkAlV0sAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TkAlV0sAnalyzer(const edm::ParameterSet&);
  ~TkAlV0sAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  template <typename T, typename... Args>
  T* book(const Args&... args) const;
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  TH1F* bookHisto1D(std::string name, std::string title, std::string xaxis, std::string yaxis, MEbinning binning);

  TH2F* bookHisto2D(std::string name,
                    std::string title,
                    std::string xaxis,
                    std::string yaxis,
                    MEbinning xbinning,
                    MEbinning ybinning);

  TProfile* bookProfile(std::string name,
                        std::string title,
                        std::string xaxis,
                        std::string yaxis,
                        MEbinning xbinning,
                        MEbinning ybinning);

  void getHistoPSet(edm::ParameterSet pset, MEbinning& mebinning);

  void fillMonitoringHistos(const edm::Event& iEvent);

  // ----------member data ---------------------------
  const edm::EDGetTokenT<TrackCollection> tracksToken_;  //used to select what tracks to read from configuration file
  const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> vccToken_;

  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::EDGetTokenT<reco::VertexCollection> pvToken_;
  const edm::EDGetTokenT<LumiScalersCollection> lumiscalersToken_;
  const edm::EDGetTokenT<OnlineLuminosityRecord> metaDataToken_;
  const bool forceSCAL_;
  const int pvNDOF_;

  // histograms
  edm::Service<TFileService> fs_;

  TH1F* h_diTrackMass;
  TH1F* h_V0Mass;

  TH1F* v0_N_;
  TH1F* v0_mass_;
  TH1F* v0_pt_;
  TH1F* v0_eta_;
  TH1F* v0_phi_;
  TH1F* v0_Lxy_;
  TH1F* v0_Lxy_wrtBS_;
  TH1F* v0_chi2oNDF_;
  TH1F* v0_deltaMass_;

  TProfile* v0_mass_vs_p_;
  TProfile* v0_mass_vs_pt_;
  TProfile* v0_mass_vs_eta_;

  TProfile* v0_deltaMass_vs_pt_;
  TProfile* v0_deltaMass_vs_eta_;

  TProfile* v0_Lxy_vs_deltaMass_;
  TProfile* v0_Lxy_vs_pt_;
  TProfile* v0_Lxy_vs_eta_;

  TH1F* n_vs_BX_;
  TProfile* v0_N_vs_BX_;
  TProfile* v0_mass_vs_BX_;
  TProfile* v0_Lxy_vs_BX_;
  TProfile* v0_deltaMass_vs_BX_;

  TH1F* n_vs_lumi_;
  TProfile* v0_N_vs_lumi_;
  TProfile* v0_mass_vs_lumi_;
  TProfile* v0_Lxy_vs_lumi_;
  TProfile* v0_deltaMass_vs_lumi_;

  TH1F* n_vs_PU_;
  TProfile* v0_N_vs_PU_;
  TProfile* v0_mass_vs_PU_;
  TProfile* v0_Lxy_vs_PU_;
  TProfile* v0_deltaMass_vs_PU_;

  TH1F* n_vs_LS_;
  TProfile* v0_N_vs_LS_;

  MEbinning mass_binning_;
  MEbinning pt_binning_;
  MEbinning eta_binning_;
  MEbinning Lxy_binning_;
  MEbinning chi2oNDF_binning_;
  MEbinning lumi_binning_;
  MEbinning pu_binning_;
  MEbinning ls_binning_;
};

static constexpr double piMass2 = 0.13957018 * 0.13957018;

//
// constructors and destructor
//
TkAlV0sAnalyzer::TkAlV0sAnalyzer(const edm::ParameterSet& iConfig)
    : tracksToken_(consumes<TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
      vccToken_(consumes<reco::VertexCompositeCandidateCollection>(
          iConfig.getParameter<edm::InputTag>("vertexCompositeCandidates"))),
      bsToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      pvToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertex"))),
      lumiscalersToken_(consumes<LumiScalersCollection>(iConfig.getParameter<edm::InputTag>("lumiScalers"))),
      metaDataToken_(consumes<OnlineLuminosityRecord>(iConfig.getParameter<edm::InputTag>("metadata"))),
      forceSCAL_(iConfig.getParameter<bool>("forceSCAL")),
      pvNDOF_(iConfig.getParameter<int>("pvNDOF")) {
  usesResource(TFileService::kSharedResource);

  v0_N_ = nullptr;
  v0_mass_ = nullptr;
  v0_pt_ = nullptr;
  v0_eta_ = nullptr;
  v0_phi_ = nullptr;
  v0_Lxy_ = nullptr;
  v0_Lxy_wrtBS_ = nullptr;
  v0_chi2oNDF_ = nullptr;
  v0_mass_vs_p_ = nullptr;
  v0_mass_vs_pt_ = nullptr;
  v0_mass_vs_eta_ = nullptr;
  v0_deltaMass_ = nullptr;
  v0_deltaMass_vs_pt_ = nullptr;
  v0_deltaMass_vs_eta_ = nullptr;

  v0_Lxy_vs_deltaMass_ = nullptr;
  v0_Lxy_vs_pt_ = nullptr;
  v0_Lxy_vs_eta_ = nullptr;

  n_vs_BX_ = nullptr;
  v0_N_vs_BX_ = nullptr;
  v0_mass_vs_BX_ = nullptr;
  v0_Lxy_vs_BX_ = nullptr;
  v0_deltaMass_vs_BX_ = nullptr;

  n_vs_lumi_ = nullptr;
  v0_N_vs_lumi_ = nullptr;
  v0_mass_vs_lumi_ = nullptr;
  v0_Lxy_vs_lumi_ = nullptr;
  v0_deltaMass_vs_lumi_ = nullptr;

  n_vs_PU_ = nullptr;
  v0_N_vs_PU_ = nullptr;
  v0_mass_vs_PU_ = nullptr;
  v0_Lxy_vs_PU_ = nullptr;
  v0_deltaMass_vs_PU_ = nullptr;

  n_vs_LS_ = nullptr;
  v0_N_vs_LS_ = nullptr;

  edm::ParameterSet histoPSet = iConfig.getParameter<edm::ParameterSet>("histoPSet");
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("massPSet"), mass_binning_);
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("ptPSet"), pt_binning_);
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("etaPSet"), eta_binning_);
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("LxyPSet"), Lxy_binning_);
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("chi2oNDFPSet"), chi2oNDF_binning_);
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("lumiPSet"), lumi_binning_);
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("puPSet"), pu_binning_);
  getHistoPSet(histoPSet.getParameter<edm::ParameterSet>("lsPSet"), ls_binning_);
}

template <typename T, typename... Args>
T* TkAlV0sAnalyzer::book(const Args&... args) const {
  T* t = fs_->make<T>(args...);
  return t;
}

void TkAlV0sAnalyzer::getHistoPSet(edm::ParameterSet pset, MEbinning& mebinning) {
  mebinning.nbins = pset.getParameter<int32_t>("nbins");
  mebinning.xmin = pset.getParameter<double>("xmin");
  mebinning.xmax = pset.getParameter<double>("xmax");
}

TH1F* TkAlV0sAnalyzer::bookHisto1D(
    std::string name, std::string title, std::string xaxis, std::string yaxis, MEbinning binning) {
  std::string title_w_axes = title + ";" + xaxis + ";" + yaxis;
  return book<TH1F>(name.c_str(), title_w_axes.c_str(), binning.nbins, binning.xmin, binning.xmax);
}

TH2F* TkAlV0sAnalyzer::bookHisto2D(
    std::string name, std::string title, std::string xaxis, std::string yaxis, MEbinning xbinning, MEbinning ybinning) {
  std::string title_w_axes = title + ";" + xaxis + ";" + yaxis;
  return book<TH2F>(name.c_str(),
                    title_w_axes.c_str(),
                    xbinning.nbins,
                    xbinning.xmin,
                    xbinning.xmax,
                    ybinning.nbins,
                    ybinning.xmin,
                    ybinning.xmax);
}

TProfile* TkAlV0sAnalyzer::bookProfile(
    std::string name, std::string title, std::string xaxis, std::string yaxis, MEbinning xbinning, MEbinning ybinning) {
  std::string title_w_axes = title + ";" + xaxis + ";" + yaxis;
  return book<TProfile>(
      name.c_str(), title_w_axes.c_str(), xbinning.nbins, xbinning.xmin, xbinning.xmax, ybinning.xmin, ybinning.xmax);
}

//
// member functions
//
// ------------ method called for each event  ------------
void TkAlV0sAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::vector<const reco::Track*> myTracks;

  edm::Handle<reco::VertexCompositeCandidateCollection> vccHandle;
  iEvent.getByToken(vccToken_, vccHandle);

  if (vccHandle->empty())
    return;

  reco::VertexCompositeCandidateCollection v0s = *vccHandle.product();

  for (const auto& track : iEvent.get(tracksToken_)) {
    myTracks.emplace_back(&track);
  }

  // exclude multiple candidates
  if (myTracks.size() != 2)
    return;

  for (const auto& v0 : v0s) {
    float mass = v0.mass();
    h_V0Mass->Fill(mass);

    for (size_t i = 0; i < v0.numberOfDaughters(); ++i) {
      //LogPrint("AlignmentTrackFromVertexCompositeCandidateSelector") << "daughter: " << i << std::endl;
      const reco::Candidate* daughter = v0.daughter(i);
      const reco::RecoChargedCandidate* chargedDaughter = dynamic_cast<const reco::RecoChargedCandidate*>(daughter);
      if (chargedDaughter) {
        //LogPrint("AlignmentTrackFromVertexCompositeCandidateSelector") << "charged daughter: " << i << std::endl;
        const reco::TrackRef trackRef = chargedDaughter->track();
        if (trackRef.isNonnull()) {
          // LogPrint("AlignmentTrackFromVertexCompositeCandidateSelector")
          // << "charged daughter has non-null trackref: " << i << std::endl;
        }
      }
    }
  }

  const auto& tplus = myTracks[0]->charge() > 0 ? myTracks[0] : myTracks[1];
  const auto& tminus = myTracks[0]->charge() < 0 ? myTracks[0] : myTracks[1];

  TLorentzVector p4_tplus(tplus->px(), tplus->py(), tplus->pz(), sqrt((tplus->p() * tplus->p()) + piMass2));
  TLorentzVector p4_tminus(tminus->px(), tminus->py(), tminus->pz(), sqrt((tminus->p() * tminus->p()) + piMass2));

  const auto& V0p4 = p4_tplus + p4_tminus;
  float track_invMass = V0p4.M();
  h_diTrackMass->Fill(track_invMass);

  fillMonitoringHistos(iEvent);
}

void TkAlV0sAnalyzer::fillMonitoringHistos(const edm::Event& iEvent) {
  size_t bx = iEvent.bunchCrossing();
  n_vs_BX_->Fill(bx);

  float lumi = -1.;
  if (forceSCAL_) {
    edm::Handle<LumiScalersCollection> lumiScalers = iEvent.getHandle(lumiscalersToken_);
    if (lumiScalers.isValid() && !lumiScalers->empty()) {
      LumiScalersCollection::const_iterator scalit = lumiScalers->begin();
      lumi = scalit->instantLumi();
    }
  } else {
    edm::Handle<OnlineLuminosityRecord> metaData = iEvent.getHandle(metaDataToken_);
    if (metaData.isValid())
      lumi = metaData->instLumi();
  }

  n_vs_lumi_->Fill(lumi);

  edm::Handle<reco::BeamSpot> beamspotHandle = iEvent.getHandle(bsToken_);
  reco::BeamSpot const* bs = nullptr;
  if (beamspotHandle.isValid())
    bs = &(*beamspotHandle);

  edm::Handle<reco::VertexCollection> pvHandle = iEvent.getHandle(pvToken_);
  reco::Vertex const* pv = nullptr;
  size_t nPV = 0;
  if (pvHandle.isValid()) {
    pv = &pvHandle->front();
    //--- pv fake (the pv collection should have size==1 and the pv==beam spot)
    if (pv->isFake() ||
        pv->tracksSize() == 0
        // definition of goodOfflinePrimaryVertex
        || pv->ndof() < pvNDOF_ || pv->z() > 24.)
      pv = nullptr;

    for (const auto& v : *pvHandle) {
      if (v.isFake())
        continue;
      if (v.ndof() < pvNDOF_)
        continue;
      if (v.z() > 24.)
        continue;
      ++nPV;
    }
  }
  n_vs_PU_->Fill(nPV);

  float nLS = static_cast<float>(iEvent.id().luminosityBlock());
  n_vs_LS_->Fill(nLS);

  edm::Handle<reco::VertexCompositeCandidateCollection> v0Handle = iEvent.getHandle(vccToken_);
  int n = (v0Handle.isValid() ? v0Handle->size() : -1);
  v0_N_->Fill(n);
  v0_N_vs_BX_->Fill(bx, n);
  v0_N_vs_lumi_->Fill(lumi, n);
  v0_N_vs_PU_->Fill(nPV, n);
  v0_N_vs_LS_->Fill(nLS, n);

  if (!v0Handle.isValid() or n == 0)
    return;

  reco::VertexCompositeCandidateCollection v0s = *v0Handle.product();
  for (const auto& v0 : v0s) {
    float mass = v0.mass();
    float pt = v0.pt();
    float p = v0.p();
    float eta = v0.eta();
    float phi = v0.phi();
    int pdgID = v0.pdgId();
    float chi2oNDF = v0.vertexNormalizedChi2();
    GlobalPoint displacementFromPV =
        (pv == nullptr ? GlobalPoint(-9., -9., 0) : GlobalPoint((pv->x() - v0.vx()), (pv->y() - v0.vy()), 0.));
    GlobalPoint displacementFromBS =
        (bs == nullptr
             ? GlobalPoint(-9., -9., 0.)
             : GlobalPoint(-1 * ((bs->position().x() - v0.vx()) + (v0.vz() - bs->position().z()) * bs->dxdz()),
                           -1 * ((bs->position().y() - v0.vy()) + (v0.vz() - bs->position().z()) * bs->dydz()),
                           0));
    float lxy = (pv == nullptr ? -9. : displacementFromPV.perp());
    float lxyWRTbs = (bs == nullptr ? -9. : displacementFromBS.perp());

    v0_mass_->Fill(mass);
    v0_pt_->Fill(pt);
    v0_eta_->Fill(eta);
    v0_phi_->Fill(phi);
    v0_Lxy_->Fill(lxy);
    v0_Lxy_wrtBS_->Fill(lxyWRTbs);
    v0_chi2oNDF_->Fill(chi2oNDF);

    v0_mass_vs_p_->Fill(p, mass);
    v0_mass_vs_pt_->Fill(pt, mass);
    v0_mass_vs_eta_->Fill(eta, mass);
    v0_mass_vs_BX_->Fill(bx, mass);
    v0_mass_vs_lumi_->Fill(lumi, mass);
    v0_mass_vs_PU_->Fill(nPV, mass);

    v0_Lxy_vs_BX_->Fill(bx, lxy);
    v0_Lxy_vs_lumi_->Fill(lumi, lxy);
    v0_Lxy_vs_PU_->Fill(nPV, lxy);

    float PDGmass = -9999.;
    switch (pdgID) {
      case 130:              // K_s
      case 310:              // K_L
        PDGmass = 0.497614;  // GeV
        break;
      case 3122:             // Lambda
      case -3122:            // Lambda
        PDGmass = 1.115683;  // GeV
        break;
      case 4122:   // Lambda_c
      case -4122:  // Lambda_c
      case 5122:   // Lambda_b
      case -5122:  // Lambda_b
      default:
        break;
    }
    float delta = (PDGmass > 0. ? (mass - PDGmass) / PDGmass : -9.);
    v0_deltaMass_->Fill(delta);
    v0_deltaMass_vs_pt_->Fill(pt, delta);
    v0_deltaMass_vs_eta_->Fill(eta, delta);
    v0_deltaMass_vs_BX_->Fill(bx, delta);
    v0_deltaMass_vs_lumi_->Fill(lumi, delta);
    v0_deltaMass_vs_PU_->Fill(nPV, delta);

    v0_Lxy_vs_deltaMass_->Fill(delta, lxy);
    v0_Lxy_vs_pt_->Fill(pt, lxy);
    v0_Lxy_vs_eta_->Fill(eta, lxy);
  }
}

void TkAlV0sAnalyzer::beginJob() {
  h_diTrackMass = book<TH1F>(
      "diTrackMass", "V0 mass from tracks in Event", mass_binning_.nbins, mass_binning_.xmin, mass_binning_.xmax);
  h_V0Mass = book<TH1F>(
      "V0kMass", "Reconstructed V0 mass in Event", mass_binning_.nbins, mass_binning_.xmin, mass_binning_.xmax);

  MEbinning N_binning;
  N_binning.nbins = 15;
  N_binning.xmin = -0.5;
  N_binning.xmax = 14.5;
  v0_N_ = bookHisto1D("v0_N", "# v0", "# v0", "events", N_binning);
  v0_mass_ = bookHisto1D("v0_mass", "mass", "mass [GeV]", "events", mass_binning_);
  v0_pt_ = bookHisto1D("v0_pt", "pt", "p_{T} [GeV]", "events", pt_binning_);
  v0_eta_ = bookHisto1D("v0_eta", "eta", "#eta", "events", eta_binning_);
  MEbinning phi_binning;
  phi_binning.nbins = 34;
  phi_binning.xmin = -3.2;
  phi_binning.xmax = 3.2;
  v0_phi_ = bookHisto1D("v0_phi", "phi", "#phi [rad]", "events", phi_binning);
  v0_Lxy_ = bookHisto1D("v0_Lxy", "Lxy", "L_{xy} w.r.t. PV [cm]", "events", Lxy_binning_);
  v0_Lxy_wrtBS_ = bookHisto1D("v0_Lxy_wrtBS", "Lxy", "L_{xy} w.r.t. BS [cm]", "events", Lxy_binning_);
  v0_chi2oNDF_ = bookHisto1D("v0_chi2oNDF", "chi2oNDF", "vertex normalized #chi^{2}", "events", chi2oNDF_binning_);

  v0_mass_vs_p_ = bookProfile("v0_mass_vs_p", "mass vs p", "p [GeV]", "mass [GeV]", pt_binning_, mass_binning_);
  v0_mass_vs_pt_ = bookProfile("v0_mass_vs_pt", "mass vs pt", "p_{T} [GeV]", "mass [GeV]", pt_binning_, mass_binning_);
  v0_mass_vs_eta_ = bookProfile("v0_mass_vs_eta", "mass vs eta", "#eta", "mass [GeV]", eta_binning_, mass_binning_);

  MEbinning delta_binning;
  delta_binning.nbins = 150;
  delta_binning.xmin = -0.15;
  delta_binning.xmax = 0.15;
  v0_deltaMass_ = bookHisto1D("v0_deltaMass", "deltaMass", "m-m_{PDG}/m_{DPG}", "events", delta_binning);
  v0_deltaMass_vs_pt_ = bookProfile(
      "v0_deltaMass_vs_pt", "deltaMass vs pt", "p_{T} [GeV]", "m-m_{PDG}/m_{DPG}", pt_binning_, delta_binning);
  v0_deltaMass_vs_eta_ =
      bookProfile("v0_deltaMass_vs_eta", "deltaMass vs eta", "#eta", "m-m_{PDG}/m_{DPG}", eta_binning_, delta_binning);

  v0_Lxy_vs_deltaMass_ = bookProfile(
      "v0_Lxy_vs_deltaMass", "L_{xy} vs deltaMass", "m-m_{PDG}/m_{DPG}", "L_{xy} [cm]", delta_binning, Lxy_binning_);
  v0_Lxy_vs_pt_ =
      bookProfile("v0_Lxy_vs_pt", "L_{xy} vs p_{T}", "p_{T} [GeV]", "L_{xy} [cm]", pt_binning_, Lxy_binning_);
  v0_Lxy_vs_eta_ = bookProfile("v0_Lxy_vs_eta", "L_{xy} vs #eta", "#eta", "L_{xy} [cm]", eta_binning_, Lxy_binning_);

  MEbinning bx_binning;
  bx_binning.nbins = 3564;
  bx_binning.xmin = 0.5;
  bx_binning.xmax = 3564.5;
  n_vs_BX_ = bookHisto1D("n_vs_BX", "# events vs BX", "BX", "# events", bx_binning);
  v0_N_vs_BX_ = bookProfile("v0_N_vs_BX", "# v0 vs BX", "BX", "# v0", bx_binning, N_binning);
  v0_mass_vs_BX_ = bookProfile("v0_mass_vs_BX", "mass vs BX", "BX", "mass [GeV]", bx_binning, mass_binning_);
  v0_Lxy_vs_BX_ = bookProfile("v0_Lxy_vs_BX", "L_{xy} vs BX", "BX", "L_{xy} [cm]", bx_binning, Lxy_binning_);
  v0_deltaMass_vs_BX_ =
      bookProfile("v0_deltaMass_vs_BX", "deltaMass vs BX", "BX", "m-m_{PDG}/m_{DPG}", bx_binning, delta_binning);

  n_vs_lumi_ =
      bookHisto1D("n_vs_lumi", "# events vs lumi", "inst. lumi x10^{30} [Hz cm^{-2}]", "# events", lumi_binning_);

  v0_N_vs_lumi_ =
      bookProfile("v0_N_vs_lumi", "# v0 vs lumi", "inst. lumi x10^{30} [Hz cm^{-2}]", "# v0", lumi_binning_, N_binning);

  v0_mass_vs_lumi_ = bookProfile(
      "v0_mass_vs_lumi", "mass vs lumi", "inst. lumi x10^{30} [Hz cm^{-2}]", "mass [GeV]", lumi_binning_, mass_binning_);

  v0_Lxy_vs_lumi_ = bookProfile("v0_Lxy_vs_lumi",
                                "L_{xy} vs lumi",
                                "inst. lumi x10^{30} [Hz cm^{-2}]",
                                "L_{xy} [cm]",
                                lumi_binning_,
                                Lxy_binning_);

  v0_deltaMass_vs_lumi_ = bookProfile("v0_deltaMass_vs_lumi",
                                      "deltaMass vs lumi",
                                      "inst. lumi x10^{30} [Hz cm^{-2}]",
                                      "m-m_{PDG}/m_{DPG}",
                                      lumi_binning_,
                                      delta_binning);

  n_vs_PU_ = bookHisto1D("n_vs_PU", "# events vs PU", "# good PV", "# events", pu_binning_);
  v0_N_vs_PU_ = bookProfile("v0_N_vs_PU", "# v0 vs PU", "# good PV", "# v0", pu_binning_, N_binning);
  v0_mass_vs_PU_ = bookProfile("v0_mass_vs_PU", "mass vs PU", "# good PV", "mass [GeV]", pu_binning_, mass_binning_);
  v0_Lxy_vs_PU_ = bookProfile("v0_Lxy_vs_PU", "L_{xy} vs PU", "# good PV", "L_{xy} [cm]", pu_binning_, Lxy_binning_);
  v0_deltaMass_vs_PU_ = bookProfile(
      "v0_deltaMass_vs_PU", "deltaMass vs PU", "# good PV", "m-m_{PDG}/m_{DPG}", pu_binning_, delta_binning);

  n_vs_LS_ = bookHisto1D("n_vs_LS", "# events vs LS", "LS", "# events", ls_binning_);
  v0_N_vs_LS_ = bookProfile("v0_N_vs_LS", "# v0 vs LS", "LS", "# v0", ls_binning_, N_binning);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TkAlV0sAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertexCompositeCandidates", edm::InputTag("generalV0Candidates:Kshort"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("ALCARECOTkAlKShortTracks"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("primaryVertex", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("lumiScalers", edm::InputTag("scalersRawToDigi"));
  desc.add<edm::InputTag>("metadata", edm::InputTag("onlineMetaDataDigis"));
  desc.add<bool>("forceSCAL", false);
  desc.add<int>("pvNDOF", 4);

  {
    edm::ParameterSetDescription psd0;
    {
      edm::ParameterSetDescription psd1;
      psd1.add<int>("nbins", 3700);
      psd1.add<double>("xmin", 0.);
      psd1.add<double>("xmax", 14000.);
      psd0.add<edm::ParameterSetDescription>("lumiPSet", psd1);
    }
    {
      edm::ParameterSetDescription psd2;
      psd2.add<int>("nbins", 100);
      psd2.add<double>("xmin", 0.400);
      psd2.add<double>("xmax", 0.600);
      psd0.add<edm::ParameterSetDescription>("massPSet", psd2);
    }
    {
      edm::ParameterSetDescription psd3;
      psd3.add<int>("nbins", 100);
      psd3.add<double>("xmin", 0.);
      psd3.add<double>("xmax", 50.);
      psd0.add<edm::ParameterSetDescription>("ptPSet", psd3);
    }
    {
      edm::ParameterSetDescription psd4;
      psd4.add<int>("nbins", 60);
      psd4.add<double>("xmin", -3.);
      psd4.add<double>("xmax", 3.);
      psd0.add<edm::ParameterSetDescription>("etaPSet", psd4);
    }
    {
      edm::ParameterSetDescription psd5;
      psd5.add<int>("nbins", 350);
      psd5.add<double>("xmin", 0.);
      psd5.add<double>("xmax", 70.);
      psd0.add<edm::ParameterSetDescription>("LxyPSet", psd5);
    }
    {
      edm::ParameterSetDescription psd6;
      psd6.add<int>("nbins", 100);
      psd6.add<double>("xmin", 0.);
      psd6.add<double>("xmax", 30.);
      psd0.add<edm::ParameterSetDescription>("chi2oNDFPSet", psd6);
    }
    {
      edm::ParameterSetDescription psd7;
      psd7.add<int>("nbins", 100);
      psd7.add<double>("xmin", -0.5);
      psd7.add<double>("xmax", 99.5);
      psd0.add<edm::ParameterSetDescription>("puPSet", psd7);
    }
    {
      edm::ParameterSetDescription psd8;
      psd8.add<int>("nbins", 2000);
      psd8.add<double>("xmin", 0.);
      psd8.add<double>("xmax", 2000.);
      psd0.add<edm::ParameterSetDescription>("lsPSet", psd8);
    }
    desc.add<edm::ParameterSetDescription>("histoPSet", psd0);
  }
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TkAlV0sAnalyzer);
-- dummy change --
