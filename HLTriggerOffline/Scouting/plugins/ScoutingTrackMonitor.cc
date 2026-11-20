// -*- C++ -*-
//
// Package:    DQMOffline/Scouting
// Class:      ScoutingTrackMonitor
//
// Description: DQM monitor for Run3 scouting tracks and vertices.
//
// Impact parameters are computed with the standard reco::TrackBase definitions
// after rebuilding a reco::Track from the scouting perigee parameters and a
// reco::Vertex from the scouting vertex. Track/vertex association is taken from
// the tk_vtxInd() stored by the scouting track packer.
//

// system includes
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <vector>
#include <fmt/format.h>

// ROOT includes
#include "TH1F.h"
#include "TMath.h"
#include "TProfile.h"

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace sctTrackMonitor {
  // same logic used for the MTV:
  // cf https://github.com/cms-sw/cmssw/blob/master/Validation/RecoTrack/src/MTVHistoProducerAlgoForTracker.cc
  typedef dqm::reco::DQMStore DQMStore;

  inline void setBinLog(TAxis* axis) {
    const int bins = axis->GetNbins();
    const float from = axis->GetXmin();
    const float to = axis->GetXmax();
    const float width = (to - from) / bins;
    std::vector<float> new_bins(bins + 1, 0);
    for (int i = 0; i <= bins; i++) {
      new_bins[i] = TMath::Power(10, from + i * width);
    }
    axis->Set(bins, new_bins.data());
  }

  inline void setBinLogX(TH1* h) { setBinLog(h->GetXaxis()); }
  inline void setBinLogY(TH1* h) { setBinLog(h->GetYaxis()); }

  template <typename... Args>
  dqm::reco::MonitorElement* makeProfileIfLog(DQMStore::IBooker& ibook, bool logx, bool logy, Args&&... args) {
    auto prof = std::make_unique<TProfile>(std::forward<Args>(args)...);
    if (logx)
      setBinLogX(prof.get());
    if (logy)
      setBinLogY(prof.get());
    const auto& name = prof->GetName();
    return ibook.bookProfile(name, prof.release());
  }

  template <typename... Args>
  dqm::reco::MonitorElement* makeTH1IfLog(DQMStore::IBooker& ibook, bool logx, bool logy, Args&&... args) {
    auto h1 = std::make_unique<TH1F>(std::forward<Args>(args)...);
    if (logx)
      setBinLogX(h1.get());
    if (logy)
      setBinLogY(h1.get());
    const auto& name = h1->GetName();
    return ibook.book1D(name, h1.release());
  }
}  // namespace sctTrackMonitor

class ScoutingTrackMonitor : public DQMEDAnalyzer {
public:
  explicit ScoutingTrackMonitor(const edm::ParameterSet&);
  ~ScoutingTrackMonitor() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // Block of impact-parameter histograms for PV tracks above a pT threshold
  // (same layout as PrimaryVertexMonitor).
  struct IPMonitoring {
    std::string varname_;
    float pTcut_ = 0.f;

    dqm::reco::MonitorElement *IP_ = nullptr, *IPErr_ = nullptr, *IPPull_ = nullptr;
    dqm::reco::MonitorElement *IPVsPhi_ = nullptr, *IPVsEta_ = nullptr, *IPVsPt_ = nullptr;
    dqm::reco::MonitorElement *IPErrVsPhi_ = nullptr, *IPErrVsEta_ = nullptr, *IPErrVsPt_ = nullptr;
    dqm::reco::MonitorElement *IPVsEtaVsPhi_ = nullptr, *IPErrVsEtaVsPhi_ = nullptr;

    void bookIPMonitor(DQMStore::IBooker&, const edm::ParameterSet&);
    void fill(float eta, float phi, float pt, float ip, float ipErr);

  private:
    int PhiBin_ = 0, EtaBin_ = 0, PtBin_ = 0;
    double PhiMin_ = 0., PhiMax_ = 0., EtaMin_ = 0., EtaMax_ = 0., PtMin_ = 0., PtMax_ = 0.;
  };

  // Generic "quantity vs eta / phi / pT" profile bundle
  struct ProfileConfig {
    std::string name;   // base name
    std::string title;  // human-readable title
    double ymin;        // minimum Y value
    double ymax;        // maximum Y value

    MonitorElement* p2_eta_phi = nullptr;
    MonitorElement* p_eta = nullptr;
    MonitorElement* p_phi = nullptr;
    MonitorElement* p_pt = nullptr;  // only booked where meaningful

    void fill(float eta, float phi, float pt, double value) const {
      p2_eta_phi->Fill(eta, phi, value);
      p_eta->Fill(eta, value);
      p_phi->Fill(phi, value);
      if (p_pt)
        p_pt->Fill(pt, value);
    }
  };

protected:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  // helpers
  reco::Track makeRecoTrack(const Run3ScoutingTrack& sTrack) const;
  reco::Vertex makeRecoVertex(const Run3ScoutingVertex& sVertex) const;

  template <typename T>
  bool getValidHandle(const edm::Event& iEvent,
                      const edm::EDGetTokenT<T>& token,
                      edm::Handle<T>& handle,
                      const std::string& label) const;

  // configuration
  const edm::ParameterSet conf_;

  // tokens
  const edm::EDGetTokenT<std::vector<Run3ScoutingTrack>> tracksToken_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> verticesToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

  const std::string topFolderName_;  // top folder name where to book histograms

  static constexpr int cmToUm = 10000;

  // ---------------- event-level ----------------
  MonitorElement* h_nTracks = nullptr;
  MonitorElement* h_nVertices = nullptr;
  MonitorElement* h_nValidVertices = nullptr;

  // ---------------- track kinematics (all tracks) ----------------
  MonitorElement* h_pt = nullptr;
  MonitorElement* h_p = nullptr;
  MonitorElement* h_eta = nullptr;
  MonitorElement* h_phi = nullptr;
  MonitorElement* h_charge = nullptr;
  MonitorElement* h2_eta_phi = nullptr;

  // ---------------- track fit quality ----------------
  MonitorElement* h_chi2 = nullptr;
  MonitorElement* h_ndof = nullptr;
  MonitorElement* h_normchi2 = nullptr;
  MonitorElement* h_chi2prob = nullptr;
  MonitorElement* h_ptErrOverPt = nullptr;

  // ---------------- perigee parameters and their errors ----------------
  MonitorElement* h_qoverp = nullptr;
  MonitorElement* h_lambda = nullptr;
  MonitorElement* h_dsz = nullptr;
  MonitorElement* h_qoverpErr = nullptr;
  MonitorElement* h_lambdaErr = nullptr;
  MonitorElement* h_phiErr = nullptr;
  MonitorElement* h_dxyErrRaw = nullptr;  // as stored in the scouting track
  MonitorElement* h_dzErrRaw = nullptr;

  // ---------------- hits ----------------
  MonitorElement* h_nPixelHits = nullptr;
  MonitorElement* h_nTrackerLayers = nullptr;
  MonitorElement* h_nStripHits = nullptr;
  std::vector<ProfileConfig> hitProfiles_;

  // ---------------- reference point ----------------
  MonitorElement* h_vr = nullptr;  // radius of the stored reference point
  MonitorElement* h_vz = nullptr;  // z of the stored reference point

  // ---------------- IP w.r.t. the origin: closure test stored vs. recomputed ----------------
  MonitorElement* h_dxyClosure = nullptr;
  MonitorElement* h_dzClosure = nullptr;
  MonitorElement* h_dxyErrClosure = nullptr;
  MonitorElement* h_dzErrClosure = nullptr;

  // ---------------- IP w.r.t. the beam spot ----------------
  MonitorElement* h_dxyBS = nullptr;
  MonitorElement* h_dzBS = nullptr;
  MonitorElement* p_dxyBS_vs_phi = nullptr;
  MonitorElement* p_dxyBS_vs_eta = nullptr;
  MonitorElement* p_dzBS_vs_phi = nullptr;
  MonitorElement* p_dzBS_vs_eta = nullptr;

  // ---------------- fit-quality profiles ----------------
  MonitorElement* p_chi2_vs_phi = nullptr;
  MonitorElement* p_chi2_vs_eta = nullptr;
  MonitorElement* p_normchi2_vs_phi = nullptr;
  MonitorElement* p_normchi2_vs_eta = nullptr;
  MonitorElement* p_normchi2_vs_pt = nullptr;
  MonitorElement* p_normchi2_vs_p = nullptr;
  MonitorElement* p_chi2Prob_vs_phi = nullptr;
  MonitorElement* p_chi2Prob_vs_eta = nullptr;
  MonitorElement* p_chi2Prob_vs_dxy = nullptr;
  MonitorElement* p_chi2Prob_vs_dz = nullptr;
  MonitorElement* p_qoverpt_vs_phi = nullptr;
  MonitorElement* p_qoverpt_vs_eta = nullptr;
  MonitorElement* p_ptRes_vs_phi = nullptr;
  MonitorElement* p_ptRes_vs_eta = nullptr;
  MonitorElement* p_ptRes_vs_pt = nullptr;

  // ---------------- track / vertex association ----------------
  MonitorElement* h_vtx_idx = nullptr;
  MonitorElement* h_trkAssoc = nullptr;
  MonitorElement* h_trk_vz_minus_pvz = nullptr;
  MonitorElement* h2_vtx_nAssoc_vs_nTracks = nullptr;

  // ---------------- IP w.r.t. the associated PV ----------------
  MonitorElement* h_dxy = nullptr;
  MonitorElement* h_dz = nullptr;
  std::vector<ProfileConfig> impactParameterProfiles_;

  IPMonitoring dxy_pt1;
  IPMonitoring dxy_pt10;
  IPMonitoring dz_pt1;
  IPMonitoring dz_pt10;

  // ---------------- vertices ----------------
  MonitorElement* h_vtx_x = nullptr;
  MonitorElement* h_vtx_y = nullptr;
  MonitorElement* h_vtx_z = nullptr;
  MonitorElement* h_vtx_xErr = nullptr;
  MonitorElement* h_vtx_yErr = nullptr;
  MonitorElement* h_vtx_zErr = nullptr;
  MonitorElement* h_vtx_xyCorr = nullptr;
  MonitorElement* h_vtx_xzCorr = nullptr;
  MonitorElement* h_vtx_yzCorr = nullptr;
  MonitorElement* h_vtx_chi2ndf = nullptr;
  MonitorElement* h_vtx_prob = nullptr;
  MonitorElement* h_vtx_ndof = nullptr;
  MonitorElement* h_vtx_nTracks = nullptr;
  MonitorElement* h_vtx_sumPt2 = nullptr;
  MonitorElement* h2_vtx_ndof_vs_nTracks = nullptr;
  MonitorElement* p_vtx_x_vs_z = nullptr;
  MonitorElement* p_vtx_y_vs_z = nullptr;

  // PV - beam spot residuals
  MonitorElement* h_pvBS_dx = nullptr;
  MonitorElement* h_pvBS_dy = nullptr;
  MonitorElement* h_pv0BS_dx = nullptr;
  MonitorElement* h_pv0BS_dy = nullptr;

  // ---------------- beam spot ----------------
  MonitorElement *h_bsX = nullptr, *h_bsY = nullptr, *h_bsZ = nullptr, *h_bsSigmaZ = nullptr;
  MonitorElement *h_bsDxdz = nullptr, *h_bsDydz = nullptr;
  MonitorElement *h_bsBeamWidthX = nullptr, *h_bsBeamWidthY = nullptr, *h_bsType = nullptr;
};

// constructor
ScoutingTrackMonitor::ScoutingTrackMonitor(const edm::ParameterSet& iConfig)
    : conf_(iConfig),
      tracksToken_{consumes<std::vector<Run3ScoutingTrack>>(iConfig.getParameter<edm::InputTag>("tracks"))},
      verticesToken_{consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("vertices"))},
      beamSpotToken_{consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotLabel"))},
      topFolderName_{iConfig.getParameter<std::string>("topFolderName")} {
  hitProfiles_ = {{"nValidPixelHits", "nValidPixelHits", 0., 10.},
                  {"nTrackerLayersWithMeasurement", "nTrackerLayersWithMeasurement", 0., 20.},
                  {"nValidStripHits", "nValidStripHits", 0., 30.}};
  impactParameterProfiles_ = {{"dxy", "d_{xy}", -0.15 * cmToUm, 0.15 * cmToUm},
                              {"dz", "d_{z}", -0.35 * cmToUm, 0.35 * cmToUm}};
}

// histogram booking
void ScoutingTrackMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  using namespace sctTrackMonitor;

  // common binning
  constexpr int nEtaBins = 50;
  constexpr double etaMin = -3.0;
  constexpr double etaMax = 3.0;
  constexpr int nPhiBins = 50;
  constexpr double phiMin = -std::numbers::pi;
  constexpr double phiMax = std::numbers::pi;
  constexpr int nPtBins = 50;
  constexpr double ptMinLog = -0.5;  // 0.3 GeV
  constexpr double ptMaxLog = 2.5;   // ~316 GeV

  const auto vposx = conf_.getParameter<double>("Xpos");
  const auto vposy = conf_.getParameter<double>("Ypos");

  // =====================================================================
  //  Event-level
  // =====================================================================
  ibooker.setCurrentFolder(topFolderName_ + "/Event");

  h_nTracks = ibooker.book1DD("nTracks", "Scouting tracks per event;N_{tracks};Events", 200, -0.5, 1999.5);
  h_nVertices = ibooker.book1DD("nVertices", "Scouting vertices per event;N_{vertices};Events", 100, -0.5, 99.5);
  h_nValidVertices =
      ibooker.book1DD("nValidVertices", "Valid scouting vertices per event;N_{valid vertices};Events", 100, -0.5, 99.5);

  // =====================================================================
  //  Tracks: kinematics, fit quality, perigee parameters, hits
  // =====================================================================
  ibooker.setCurrentFolder(topFolderName_ + "/Tracks");

  h_pt = makeTH1IfLog(ibooker, true, false, "pt", "Track p_{T};p_{T} [GeV];Tracks", nPtBins, ptMinLog, ptMaxLog);
  h_p = makeTH1IfLog(ibooker, true, false, "p", "Track p;p [GeV];Tracks", nPtBins, ptMinLog, 3.0);
  h_eta = ibooker.book1DD("eta", "Track #eta;#eta;Tracks", 60, etaMin, etaMax);
  h_phi = ibooker.book1DD("phi", "Track #phi;#phi [rad];Tracks", 64, phiMin, phiMax);
  h_charge = ibooker.book1DD("charge", "Track charge;q;Tracks", 3, -1.5, 1.5);
  h2_eta_phi = ibooker.book2I(
      "eta_vs_phi", "Track occupancy;#eta;#phi [rad]", nEtaBins, etaMin, etaMax, nPhiBins, phiMin, phiMax);
  h2_eta_phi->setOption("colz");

  h_chi2 = ibooker.book1DD("chi2", "Track #chi^{2};#chi^{2};Tracks", 100, 0., 100.);
  h_ndof = ibooker.book1DD("ndof", "Track ndof;ndof;Tracks", 60, -0.5, 59.5);
  h_normchi2 = ibooker.book1DD("normchi2", "Track #chi^{2}/ndof;#chi^{2}/ndof;Tracks", 100, 0., 10.);
  h_chi2prob = ibooker.book1DD("chi2prob", "Track #chi^{2} probability;P(#chi^{2},ndof);Tracks", 100, 0., 1.);
  h_ptErrOverPt = ibooker.book1DD("ptErrOverPt", "Track #sigma(p_{T})/p_{T};#sigma(p_{T})/p_{T};Tracks", 100, 0., 0.5);

  h_qoverp = ibooker.book1DD("qoverp", "Track q/p;q/p [GeV^{-1}];Tracks", 100, -5., 5.);
  h_lambda = ibooker.book1DD("lambda", "Track #lambda;#lambda [rad];Tracks", 100, -1.6, 1.6);
  h_dsz = ibooker.book1DD("dsz", "Track d_{sz} (w.r.t. origin);d_{sz} [cm];Tracks", 100, -30., 30.);
  h_qoverpErr = ibooker.book1DD("qoverpErr", "Track #sigma(q/p);#sigma(q/p) [GeV^{-1}];Tracks", 100, 0., 0.1);
  h_lambdaErr = ibooker.book1DD("lambdaErr", "Track #sigma(#lambda);#sigma(#lambda) [rad];Tracks", 100, 0., 0.01);
  h_phiErr = ibooker.book1DD("phiErr", "Track #sigma(#phi);#sigma(#phi) [rad];Tracks", 100, 0., 0.01);
  h_dxyErrRaw =
      ibooker.book1DD("dxyErrStored", "Stored track #sigma(d_{xy});#sigma(d_{xy}) [#mum];Tracks", 100, 0., 2000.);
  h_dzErrRaw = ibooker.book1DD("dzErrStored", "Stored track #sigma(d_{z});#sigma(d_{z}) [#mum];Tracks", 100, 0., 5000.);

  h_nPixelHits = ibooker.book1DD("nValidPixelHits", "Valid pixel hits;N_{pixel hits};Tracks", 11, -0.5, 10.5);
  h_nTrackerLayers = ibooker.book1DD(
      "nTrackerLayersWithMeasurement", "Tracker layers with measurement;N_{layers};Tracks", 21, -0.5, 20.5);
  h_nStripHits = ibooker.book1DD("nValidStripHits", "Valid strip hits;N_{strip hits};Tracks", 31, -0.5, 30.5);

  h_vr = ibooker.book1DD("vr", "Reference point radius;r_{ref} [cm];Tracks", 100, 0., 1.);
  h_vz = ibooker.book1DD("vz", "Reference point z;z_{ref} [cm];Tracks", 100, -30., 30.);

  // hit profiles vs eta / phi / pT
  for (auto& cfg : hitProfiles_) {
    const std::string& base = cfg.name;
    const std::string& title = cfg.title;

    cfg.p2_eta_phi = ibooker.bookProfile2D(base + "_vs_eta_phi_prof",
                                           title + " vs #eta-#phi;#eta;#phi [rad];#LT" + title + "#GT",
                                           nEtaBins,
                                           etaMin,
                                           etaMax,
                                           nPhiBins,
                                           phiMin,
                                           phiMax,
                                           cfg.ymin,
                                           cfg.ymax,
                                           "");
    cfg.p2_eta_phi->setOption("colz");

    cfg.p_eta = ibooker.bookProfile(base + "_vs_eta_prof",
                                    title + " vs #eta;#eta;#LT" + title + "#GT",
                                    nEtaBins,
                                    etaMin,
                                    etaMax,
                                    cfg.ymin,
                                    cfg.ymax,
                                    "");

    cfg.p_phi = ibooker.bookProfile(base + "_vs_phi_prof",
                                    title + " vs #phi;#phi [rad];#LT" + title + "#GT",
                                    nPhiBins,
                                    phiMin,
                                    phiMax,
                                    cfg.ymin,
                                    cfg.ymax,
                                    "");

    cfg.p_pt = makeProfileIfLog(ibooker,
                                true,
                                false,
                                (base + "_vs_pt_prof").c_str(),
                                (title + " vs p_{T};p_{T} [GeV];#LT" + title + "#GT").c_str(),
                                nPtBins,
                                ptMinLog,
                                ptMaxLog,
                                cfg.ymin,
                                cfg.ymax,
                                "");
  }

  // fit-quality profiles
  p_chi2_vs_phi = ibooker.bookProfile(
      "p_chi2_vs_phi", "#chi^{2} vs #phi;#phi [rad];#LT#chi^{2}#GT", nPhiBins, phiMin, phiMax, 0., 100., "");
  p_chi2_vs_eta = ibooker.bookProfile(
      "p_chi2_vs_eta", "#chi^{2} vs #eta;#eta;#LT#chi^{2}#GT", nEtaBins, etaMin, etaMax, 0., 100., "");
  p_normchi2_vs_phi = ibooker.bookProfile("p_normchi2_vs_phi",
                                          "#chi^{2}/ndof vs #phi;#phi [rad];#LT#chi^{2}/ndof#GT",
                                          nPhiBins,
                                          phiMin,
                                          phiMax,
                                          0.,
                                          10.,
                                          "");
  p_normchi2_vs_eta = ibooker.bookProfile(
      "p_normchi2_vs_eta", "#chi^{2}/ndof vs #eta;#eta;#LT#chi^{2}/ndof#GT", nEtaBins, etaMin, etaMax, 0., 10., "");
  p_normchi2_vs_pt = makeProfileIfLog(ibooker,
                                      true,
                                      false,
                                      "p_normchi2_vs_pt",
                                      "#chi^{2}/ndof vs p_{T};p_{T} [GeV];#LT#chi^{2}/ndof#GT",
                                      nPtBins,
                                      ptMinLog,
                                      ptMaxLog,
                                      0.,
                                      10.,
                                      "");
  p_normchi2_vs_p = makeProfileIfLog(ibooker,
                                     true,
                                     false,
                                     "p_normchi2_vs_p",
                                     "#chi^{2}/ndof vs p;p [GeV];#LT#chi^{2}/ndof#GT",
                                     nPtBins,
                                     ptMinLog,
                                     3.0,
                                     0.,
                                     10.,
                                     "");
  p_chi2Prob_vs_phi = ibooker.bookProfile("p_chi2Prob_vs_phi",
                                          "#chi^{2} probability vs #phi;#phi [rad];#LT P(#chi^{2}) #GT",
                                          nPhiBins,
                                          phiMin,
                                          phiMax,
                                          0.,
                                          1.,
                                          "");
  p_chi2Prob_vs_eta = ibooker.bookProfile("p_chi2Prob_vs_eta",
                                          "#chi^{2} probability vs #eta;#eta;#LT P(#chi^{2}) #GT",
                                          nEtaBins,
                                          etaMin,
                                          etaMax,
                                          0.,
                                          1.,
                                          "");
  p_chi2Prob_vs_dxy =
      ibooker.bookProfile("p_chi2Prob_vs_dxy",
                          "#chi^{2} probability vs |d_{xy}(BS)|;|d_{xy}(BS)| [#mum];#LT P(#chi^{2}) #GT",
                          50,
                          0.,
                          0.15 * cmToUm,
                          0.,
                          1.,
                          "");
  p_chi2Prob_vs_dz = ibooker.bookProfile("p_chi2Prob_vs_dz",
                                         "#chi^{2} probability vs d_{z}(BS);d_{z}(BS) [cm];#LT P(#chi^{2}) #GT",
                                         50,
                                         -20.,
                                         20.,
                                         0.,
                                         1.,
                                         "");
  p_qoverpt_vs_phi = ibooker.bookProfile("p_qoverpt_vs_phi",
                                         "q/p_{T} vs #phi;#phi [rad];#LT q/p_{T} #GT [GeV^{-1}]",
                                         nPhiBins,
                                         phiMin,
                                         phiMax,
                                         -5.,
                                         5.,
                                         "");
  p_qoverpt_vs_eta = ibooker.bookProfile(
      "p_qoverpt_vs_eta", "q/p_{T} vs #eta;#eta;#LT q/p_{T} #GT [GeV^{-1}]", nEtaBins, etaMin, etaMax, -5., 5., "");
  p_ptRes_vs_phi = ibooker.bookProfile("p_ptResolution_vs_phi",
                                       "#sigma(p_{T})/p_{T} vs #phi;#phi [rad];#LT#sigma(p_{T})/p_{T}#GT",
                                       nPhiBins,
                                       phiMin,
                                       phiMax,
                                       0.,
                                       1.,
                                       "");
  p_ptRes_vs_eta = ibooker.bookProfile("p_ptResolution_vs_eta",
                                       "#sigma(p_{T})/p_{T} vs #eta;#eta;#LT#sigma(p_{T})/p_{T}#GT",
                                       nEtaBins,
                                       etaMin,
                                       etaMax,
                                       0.,
                                       1.,
                                       "");
  p_ptRes_vs_pt = makeProfileIfLog(ibooker,
                                   true,
                                   false,
                                   "p_ptResolution_vs_pt",
                                   "#sigma(p_{T})/p_{T} vs p_{T};p_{T} [GeV];#LT#sigma(p_{T})/p_{T}#GT",
                                   nPtBins,
                                   ptMinLog,
                                   ptMaxLog,
                                   0.,
                                   1.,
                                   "");

  // =====================================================================
  //  Tracks: impact parameters w.r.t. beam spot and closure tests
  // =====================================================================
  ibooker.setCurrentFolder(topFolderName_ + "/TracksBS");

  h_dxyBS =
      ibooker.book1DD("dxyBS", "d_{xy} w.r.t. beam spot;d_{xy}(BS) [#mum];Tracks", 100, -0.15 * cmToUm, 0.15 * cmToUm);
  h_dzBS = ibooker.book1DD("dzBS", "d_{z} w.r.t. beam spot;d_{z}(BS) [cm];Tracks", 100, -20., 20.);
  p_dxyBS_vs_phi = ibooker.bookProfile("p_dxyBS_vs_phi",
                                       "d_{xy}(BS) vs #phi;#phi [rad];#LT d_{xy}(BS) #GT [#mum]",
                                       nPhiBins,
                                       phiMin,
                                       phiMax,
                                       -0.15 * cmToUm,
                                       0.15 * cmToUm,
                                       "");
  p_dxyBS_vs_eta = ibooker.bookProfile("p_dxyBS_vs_eta",
                                       "d_{xy}(BS) vs #eta;#eta;#LT d_{xy}(BS) #GT [#mum]",
                                       nEtaBins,
                                       etaMin,
                                       etaMax,
                                       -0.15 * cmToUm,
                                       0.15 * cmToUm,
                                       "");
  p_dzBS_vs_phi = ibooker.bookProfile(
      "p_dzBS_vs_phi", "d_{z}(BS) vs #phi;#phi [rad];#LT d_{z}(BS) #GT [cm]", nPhiBins, phiMin, phiMax, -20., 20., "");
  p_dzBS_vs_eta = ibooker.bookProfile(
      "p_dzBS_vs_eta", "d_{z}(BS) vs #eta;#eta;#LT d_{z}(BS) #GT [cm]", nEtaBins, etaMin, etaMax, -20., 20., "");

  // closure: reco::Track rebuilt from the perigee parameters vs. the values stored by the packer.
  // Both are w.r.t. the origin; any non-zero residual indicates a change in the packer's reference point.
  h_dxyClosure =
      ibooker.book1DD("dxyClosure", "d_{xy}(rebuilt) - d_{xy}(stored);#Delta d_{xy} [#mum];Tracks", 100, -1., 1.);
  h_dzClosure = ibooker.book1DD("dzClosure", "d_{z}(rebuilt) - d_{z}(stored);#Delta d_{z} [#mum];Tracks", 100, -1., 1.);
  h_dxyErrClosure =
      ibooker.book1DD("dxyErrClosure",
                      "#sigma(d_{xy})(rebuilt) - #sigma(d_{xy})(stored);#Delta#sigma(d_{xy}) [#mum];Tracks",
                      100,
                      -1.,
                      1.);
  h_dzErrClosure = ibooker.book1DD(
      "dzErrClosure", "#sigma(d_{z})(rebuilt) - #sigma(d_{z})(stored);#Delta#sigma(d_{z}) [#mum];Tracks", 100, -1., 1.);

  // =====================================================================
  //  Tracks: association to the PV and impact parameters w.r.t. the PV
  // =====================================================================
  ibooker.setCurrentFolder(topFolderName_ + "/TracksPV");

  h_vtx_idx = ibooker.book1DD("vertexIndex", "Track vertex index;Vertex index;Tracks", 52, -1.5, 50.5);
  h_trkAssoc = ibooker.book1DD("vertexAssociation", "Track-vertex association;;Tracks", 3, -0.5, 2.5);
  h_trkAssoc->setBinLabel(1, "no vertex");
  h_trkAssoc->setBinLabel(2, "leading PV");
  h_trkAssoc->setBinLabel(3, "other PV");
  h_trk_vz_minus_pvz =
      ibooker.book1DD("trkVzMinusPVz", "z_{ref}(track) - z(PV);z_{ref} - z_{PV} [cm];Tracks", 100, -0.5, 0.5);

  h_dxy = ibooker.book1DD("dxy", "d_{xy} w.r.t. PV;d_{xy} [#mum];Tracks", 100, -0.15 * cmToUm, 0.15 * cmToUm);
  h_dz = ibooker.book1DD("dz", "d_{z} w.r.t. PV;d_{z} [#mum];Tracks", 100, -0.35 * cmToUm, 0.35 * cmToUm);

  for (auto& cfg : impactParameterProfiles_) {
    cfg.p_eta = ibooker.bookProfile(cfg.name + "_vs_eta",
                                    cfg.title + " vs #eta;#eta;#LT" + cfg.title + "#GT [#mum]",
                                    nEtaBins,
                                    etaMin,
                                    etaMax,
                                    cfg.ymin,
                                    cfg.ymax,
                                    "");
    cfg.p_phi = ibooker.bookProfile(cfg.name + "_vs_phi",
                                    cfg.title + " vs #phi;#phi [rad];#LT" + cfg.title + "#GT [#mum]",
                                    nPhiBins,
                                    phiMin,
                                    phiMax,
                                    cfg.ymin,
                                    cfg.ymax,
                                    "");
    cfg.p2_eta_phi = ibooker.bookProfile2D(cfg.name + "_vs_eta_phi",
                                           cfg.title + " vs #eta-#phi;#eta;#phi [rad];#LT" + cfg.title + "#GT [#mum]",
                                           nEtaBins,
                                           etaMin,
                                           etaMax,
                                           nPhiBins,
                                           phiMin,
                                           phiMax,
                                           cfg.ymin,
                                           cfg.ymax,
                                           "");
    cfg.p2_eta_phi->setOption("colz");
    // vs pT is covered by the IPMonitoring blocks
  }

  dxy_pt1.varname_ = "xy";
  dxy_pt1.pTcut_ = 1.f;
  dxy_pt1.bookIPMonitor(ibooker, conf_);

  dxy_pt10.varname_ = "xy";
  dxy_pt10.pTcut_ = 10.f;
  dxy_pt10.bookIPMonitor(ibooker, conf_);

  dz_pt1.varname_ = "z";
  dz_pt1.pTcut_ = 1.f;
  dz_pt1.bookIPMonitor(ibooker, conf_);

  dz_pt10.varname_ = "z";
  dz_pt10.pTcut_ = 10.f;
  dz_pt10.bookIPMonitor(ibooker, conf_);

  // =====================================================================
  //  Vertices
  // =====================================================================
  ibooker.setCurrentFolder(topFolderName_ + "/Vertices");

  h_vtx_x = ibooker.book1DD("vtxX", "Vertex x;x [cm];Vertices", 100, vposx - 0.1, vposx + 0.1);
  h_vtx_y = ibooker.book1DD("vtxY", "Vertex y;y [cm];Vertices", 100, vposy - 0.1, vposy + 0.1);
  h_vtx_z = ibooker.book1DD("vtxZ", "Vertex z;z [cm];Vertices", 100, -20., 20.);
  h_vtx_xErr = ibooker.book1DD("vtxXErr", "Vertex #sigma(x);#sigma(x) [#mum];Vertices", 100, 0., 500.);
  h_vtx_yErr = ibooker.book1DD("vtxYErr", "Vertex #sigma(y);#sigma(y) [#mum];Vertices", 100, 0., 500.);
  h_vtx_zErr = ibooker.book1DD("vtxZErr", "Vertex #sigma(z);#sigma(z) [#mum];Vertices", 100, 0., 1000.);
  h_vtx_xyCorr = ibooker.book1DD("vtxXYCorr", "Vertex #rho(x,y);#rho(x,y);Vertices", 100, -1., 1.);
  h_vtx_xzCorr = ibooker.book1DD("vtxXZCorr", "Vertex #rho(x,z);#rho(x,z);Vertices", 100, -1., 1.);
  h_vtx_yzCorr = ibooker.book1DD("vtxYZCorr", "Vertex #rho(y,z);#rho(y,z);Vertices", 100, -1., 1.);

  h_vtx_chi2ndf = ibooker.book1DD("vtxChi2ndf", "Vertex #chi^{2}/ndof;#chi^{2}/ndof;Vertices", 100, 0., 20.);
  h_vtx_prob = ibooker.book1DD("vtxChi2prob", "Vertex #chi^{2} probability;P(#chi^{2},ndof);Vertices", 100, 0., 1.);
  h_vtx_ndof = ibooker.book1DD("vtxNdof", "Vertex ndof;ndof;Vertices", 100, -0.5, 199.5);
  h_vtx_nTracks = ibooker.book1DD("vtxNTracks", "Tracks per vertex;N_{tracks};Vertices", 100, -0.5, 199.5);
  h_vtx_sumPt2 = ibooker.book1DD(
      "vtxSumPt2", "Vertex #Sigma p_{T}^{2} (associated tracks);#Sigma p_{T}^{2} [GeV^{2}];Vertices", 100, 0., 10000.);
  h2_vtx_ndof_vs_nTracks = ibooker.book2D(
      "vtxNdof_vs_nTracks", "Vertex ndof vs tracksSize;tracksSize;ndof", 50, -0.5, 199.5, 50, -0.5, 199.5);
  h2_vtx_ndof_vs_nTracks->setOption("colz");
  h2_vtx_nAssoc_vs_nTracks = ibooker.book2D("vtxNAssoc_vs_nTracks",
                                            "Tracks associated via tk_vtxInd vs tracksSize;tracksSize;N_{assoc}",
                                            50,
                                            -0.5,
                                            199.5,
                                            50,
                                            -0.5,
                                            199.5);
  h2_vtx_nAssoc_vs_nTracks->setOption("colz");

  p_vtx_x_vs_z = ibooker.bookProfile(
      "p_vtxX_vs_z", "Vertex x vs z;z [cm];#LT x #GT [cm]", 40, -20., 20., vposx - 0.1, vposx + 0.1, "");
  p_vtx_y_vs_z = ibooker.bookProfile(
      "p_vtxY_vs_z", "Vertex y vs z;z [cm];#LT y #GT [cm]", 40, -20., 20., vposy - 0.1, vposy + 0.1, "");

  h_pvBS_dx = ibooker.book1DD(
      "pvBS_dx", "x(PV) - x(BS at z_{PV}), all valid vertices;#Delta x [#mum];Vertices", 100, -300., 300.);
  h_pvBS_dy = ibooker.book1DD(
      "pvBS_dy", "y(PV) - y(BS at z_{PV}), all valid vertices;#Delta y [#mum];Vertices", 100, -300., 300.);
  h_pv0BS_dx = ibooker.book1DD("pv0BS_dx", "x(PV_{0}) - x(BS at z_{PV});#Delta x [#mum];Events", 100, -300., 300.);
  h_pv0BS_dy = ibooker.book1DD("pv0BS_dy", "y(PV_{0}) - y(BS at z_{PV});#Delta y [#mum];Events", 100, -300., 300.);

  // =====================================================================
  //  Beam spot
  // =====================================================================
  ibooker.setCurrentFolder(topFolderName_ + "/BeamSpot");

  h_bsX = ibooker.book1D("bsX", "BeamSpot x0;x_{0} [cm];Events", 100, vposx - 0.1, vposx + 0.1);
  h_bsY = ibooker.book1D("bsY", "BeamSpot y0;y_{0} [cm];Events", 100, vposy - 0.1, vposy + 0.1);
  h_bsZ = ibooker.book1D("bsZ", "BeamSpot z0;z_{0} [cm];Events", 100, -2., 2.);
  h_bsSigmaZ = ibooker.book1D("bsSigmaZ", "BeamSpot sigmaZ;#sigma_{z} [cm];Events", 100, 0., 10.);
  h_bsDxdz = ibooker.book1D("bsDxdz", "BeamSpot dxdz;dx/dz;Events", 100, -0.0003, 0.0003);
  h_bsDydz = ibooker.book1D("bsDydz", "BeamSpot dydz;dy/dz;Events", 100, -0.0003, 0.0003);
  h_bsBeamWidthX = ibooker.book1D("bsBeamWidthX", "BeamSpot BeamWidthX;width_{x} [#mum];Events", 500, 0., 15.);
  h_bsBeamWidthY = ibooker.book1D("bsBeamWidthY", "BeamSpot BeamWidthY;width_{y} [#mum];Events", 500, 0., 15.);
  h_bsType = ibooker.book1D("bsType", "BeamSpot type", 4, -1.5, 2.5);
  h_bsType->setBinLabel(1, "Unknown");
  h_bsType->setBinLabel(2, "Fake");
  h_bsType->setBinLabel(3, "LHC");
  h_bsType->setBinLabel(4, "Tracker");
}

void ScoutingTrackMonitor::IPMonitoring::bookIPMonitor(DQMStore::IBooker& iBooker, const edm::ParameterSet& config) {
  const int VarBin = config.getParameter<int>(fmt::format("D{}Bin", varname_));
  const double VarMin = config.getParameter<double>(fmt::format("D{}Min", varname_));
  const double VarMax = config.getParameter<double>(fmt::format("D{}Max", varname_));

  PhiBin_ = config.getParameter<int>("PhiBin");
  PhiMin_ = config.getParameter<double>("PhiMin");
  PhiMax_ = config.getParameter<double>("PhiMax");
  const int PhiBin2D = config.getParameter<int>("PhiBin2D");

  EtaBin_ = config.getParameter<int>("EtaBin");
  EtaMin_ = config.getParameter<double>("EtaMin");
  EtaMax_ = config.getParameter<double>("EtaMax");
  const int EtaBin2D = config.getParameter<int>("EtaBin2D");

  PtBin_ = config.getParameter<int>("PtBin");
  PtMin_ = config.getParameter<double>("PtMin") * pTcut_;
  PtMax_ = config.getParameter<double>("PtMax") * pTcut_;

  // One error range for the 1D histogram *and* all error profiles, so that
  // no profile silently drops entries outside its y-range.
  const double errMax = (varname_ == "xy") ? 2000. : 10000.;

  // 1D variables
  IP_ = iBooker.book1DD(fmt::format("d{}_pt{}", varname_, pTcut_),
                        fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_),
                        VarBin,
                        VarMin,
                        VarMax);

  IPErr_ = iBooker.book1DD(fmt::format("d{}Err_pt{}", varname_, pTcut_),
                           fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_),
                           100,
                           0.,
                           errMax);

  IPPull_ = iBooker.book1DD(
      fmt::format("d{}Pull_pt{}", varname_, pTcut_),
      fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}}/#sigma_{{d_{{{}}}}}", pTcut_, varname_, varname_),
      100,
      -5.,
      5.);

  // IP profiles
  IPVsPhi_ = iBooker.bookProfile(fmt::format("d{}VsPhi_pt{}", varname_, pTcut_),
                                 fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track #phi", pTcut_, varname_),
                                 PhiBin_,
                                 PhiMin_,
                                 PhiMax_,
                                 VarBin,
                                 VarMin,
                                 VarMax,
                                 "");
  IPVsPhi_->setAxisTitle(fmt::format("PV track (p_{{T}} > {} GeV) #phi", pTcut_), 1);
  IPVsPhi_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_), 2);

  IPVsEta_ = iBooker.bookProfile(fmt::format("d{}VsEta_pt{}", varname_, pTcut_),
                                 fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track #eta", pTcut_, varname_),
                                 EtaBin_,
                                 EtaMin_,
                                 EtaMax_,
                                 VarBin,
                                 VarMin,
                                 VarMax,
                                 "");
  IPVsEta_->setAxisTitle(fmt::format("PV track (p_{{T}} > {} GeV) #eta", pTcut_), 1);
  IPVsEta_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_), 2);

  IPVsPt_ = sctTrackMonitor::makeProfileIfLog(
      iBooker,
      true,  /* x-axis */
      false, /* y-axis */
      fmt::format("d{}VsPt_pt{}", varname_, pTcut_).c_str(),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track p_{{T}}", pTcut_, varname_).c_str(),
      PtBin_,
      std::log10(PtMin_),
      std::log10(PtMax_),
      VarMin,
      VarMax,
      "");
  IPVsPt_->setAxisTitle(fmt::format("PV track (p_{{T}} > {} GeV) p_{{T}} [GeV]", pTcut_), 1);
  IPVsPt_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_), 2);

  // IP error profiles
  IPErrVsPhi_ =
      iBooker.bookProfile(fmt::format("d{}ErrVsPhi_pt{}", varname_, pTcut_),
                          fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #phi", pTcut_, varname_),
                          PhiBin_,
                          PhiMin_,
                          PhiMax_,
                          100,
                          0.,
                          errMax,
                          "");
  IPErrVsPhi_->setAxisTitle(fmt::format("PV track (p_{{T}} > {} GeV) #phi", pTcut_), 1);
  IPErrVsPhi_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_), 2);

  IPErrVsEta_ =
      iBooker.bookProfile(fmt::format("d{}ErrVsEta_pt{}", varname_, pTcut_),
                          fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #eta", pTcut_, varname_),
                          EtaBin_,
                          EtaMin_,
                          EtaMax_,
                          100,
                          0.,
                          errMax,
                          "");
  IPErrVsEta_->setAxisTitle(fmt::format("PV track (p_{{T}} > {} GeV) #eta", pTcut_), 1);
  IPErrVsEta_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_), 2);

  IPErrVsPt_ = sctTrackMonitor::makeProfileIfLog(
      iBooker,
      true,  /* x-axis */
      false, /* y-axis */
      fmt::format("d{}ErrVsPt_pt{}", varname_, pTcut_).c_str(),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track p_{{T}}", pTcut_, varname_).c_str(),
      PtBin_,
      std::log10(PtMin_),
      std::log10(PtMax_),
      0.,
      errMax,
      "");
  IPErrVsPt_->setAxisTitle(fmt::format("PV track (p_{{T}} > {} GeV) p_{{T}} [GeV]", pTcut_), 1);
  IPErrVsPt_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_), 2);

  // 2D profiles
  IPVsEtaVsPhi_ = iBooker.bookProfile2D(
      fmt::format("d{}VsEtaVsPhi_pt{}", varname_, pTcut_),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track #eta VS track #phi", pTcut_, varname_),
      EtaBin2D,
      EtaMin_,
      EtaMax_,
      PhiBin2D,
      PhiMin_,
      PhiMax_,
      VarBin,
      VarMin,
      VarMax,
      "");
  IPVsEtaVsPhi_->setAxisTitle(fmt::format("PV track (p_{{T}} > {} GeV) #eta", pTcut_), 1);
  IPVsEtaVsPhi_->setAxisTitle(fmt::format("PV track (p_{{T}} > {} GeV) #phi", pTcut_), 2);
  IPVsEtaVsPhi_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_), 3);
  IPVsEtaVsPhi_->setOption("colz");

  IPErrVsEtaVsPhi_ = iBooker.bookProfile2D(
      fmt::format("d{}ErrVsEtaVsPhi_pt{}", varname_, pTcut_),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #eta VS track #phi", pTcut_, varname_),
      EtaBin2D,
      EtaMin_,
      EtaMax_,
      PhiBin2D,
      PhiMin_,
      PhiMax_,
      100,
      0.,
      errMax,
      "");
  IPErrVsEtaVsPhi_->setAxisTitle(fmt::format("PV track (p_{{T}} > {} GeV) #eta", pTcut_), 1);
  IPErrVsEtaVsPhi_->setAxisTitle(fmt::format("PV track (p_{{T}} > {} GeV) #phi", pTcut_), 2);
  IPErrVsEtaVsPhi_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_),
                                 3);
  IPErrVsEtaVsPhi_->setOption("colz");
}

void ScoutingTrackMonitor::IPMonitoring::fill(float eta, float phi, float pt, float ip, float ipErr) {
  IP_->Fill(ip);
  IPVsPhi_->Fill(phi, ip);
  IPVsEta_->Fill(eta, ip);
  IPVsPt_->Fill(pt, ip);
  IPVsEtaVsPhi_->Fill(eta, phi, ip);

  IPErr_->Fill(ipErr);
  IPErrVsPhi_->Fill(phi, ipErr);
  IPErrVsEta_->Fill(eta, ipErr);
  IPErrVsPt_->Fill(pt, ipErr);
  IPErrVsEtaVsPhi_->Fill(eta, phi, ipErr);

  if (ipErr > 0.f)
    IPPull_->Fill(ip / ipErr);
}

template <typename T>
bool ScoutingTrackMonitor::getValidHandle(const edm::Event& iEvent,
                                          const edm::EDGetTokenT<T>& token,
                                          edm::Handle<T>& handle,
                                          const std::string& label) const {
  iEvent.getByToken(token, handle);
  if (!handle.isValid()) {
    edm::LogWarning("ScoutingTrackMonitor") << "Invalid handle for " << label;
    return false;
  }
  return true;
}

// main event loop
void ScoutingTrackMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<std::vector<Run3ScoutingVertex>> primaryVerticesH;
  edm::Handle<std::vector<Run3ScoutingTrack>> tracksH;

  if (!getValidHandle(iEvent, verticesToken_, primaryVerticesH, "primary vertices") ||
      !getValidHandle(iEvent, tracksToken_, tracksH, "tracks")) {
    return;
  }

  // dereference handles when it's safe to do so.
  auto const& tracks = *tracksH;
  auto const& vertices = *primaryVerticesH;

  // -------------------------------------------------------------------
  // Beam spot (optional: everything that needs it is skipped if missing)
  // -------------------------------------------------------------------
  const reco::BeamSpot* beamSpot = nullptr;
  edm::Handle<reco::BeamSpot> beamSpotH;
  if (getValidHandle(iEvent, beamSpotToken_, beamSpotH, "beamSpot")) {
    beamSpot = &(*beamSpotH);
    h_bsX->Fill(beamSpot->x0());
    h_bsY->Fill(beamSpot->y0());
    h_bsZ->Fill(beamSpot->z0());
    h_bsSigmaZ->Fill(beamSpot->sigmaZ());
    h_bsDxdz->Fill(beamSpot->dxdz());
    h_bsDydz->Fill(beamSpot->dydz());
    h_bsBeamWidthX->Fill(beamSpot->BeamWidthX() * cmToUm);
    h_bsBeamWidthY->Fill(beamSpot->BeamWidthY() * cmToUm);
    h_bsType->Fill(beamSpot->type());
  }

  // -------------------------------------------------------------------
  // Event-level counts
  // -------------------------------------------------------------------
  const unsigned int nVtx = vertices.size();
  h_nTracks->Fill(tracks.size());
  h_nVertices->Fill(nVtx);

  // -------------------------------------------------------------------
  // Vertex loop
  // -------------------------------------------------------------------
  unsigned int nValidVtx = 0;
  for (unsigned int iv = 0; iv < nVtx; ++iv) {
    const auto& vtx = vertices[iv];
    if (!vtx.isValidVtx())
      continue;
    ++nValidVtx;

    h_vtx_x->Fill(vtx.x());
    h_vtx_y->Fill(vtx.y());
    h_vtx_z->Fill(vtx.z());
    h_vtx_xErr->Fill(vtx.xError() * cmToUm);
    h_vtx_yErr->Fill(vtx.yError() * cmToUm);
    h_vtx_zErr->Fill(vtx.zError() * cmToUm);

    if (vtx.xError() > 0.f && vtx.yError() > 0.f)
      h_vtx_xyCorr->Fill(vtx.xyCov() / (vtx.xError() * vtx.yError()));
    if (vtx.xError() > 0.f && vtx.zError() > 0.f)
      h_vtx_xzCorr->Fill(vtx.xzCov() / (vtx.xError() * vtx.zError()));
    if (vtx.yError() > 0.f && vtx.zError() > 0.f)
      h_vtx_yzCorr->Fill(vtx.yzCov() / (vtx.yError() * vtx.zError()));

    if (vtx.ndof() > 0) {
      h_vtx_chi2ndf->Fill(vtx.chi2() / vtx.ndof());
      h_vtx_prob->Fill(TMath::Prob(vtx.chi2(), vtx.ndof()));
    }
    h_vtx_ndof->Fill(vtx.ndof());
    h_vtx_nTracks->Fill(vtx.tracksSize());
    h2_vtx_ndof_vs_nTracks->Fill(vtx.tracksSize(), vtx.ndof());

    p_vtx_x_vs_z->Fill(vtx.z(), vtx.x());
    p_vtx_y_vs_z->Fill(vtx.z(), vtx.y());

    if (beamSpot) {
      const auto bsAtZ = beamSpot->position(vtx.z());
      const double dx = (vtx.x() - bsAtZ.x()) * cmToUm;
      const double dy = (vtx.y() - bsAtZ.y()) * cmToUm;
      h_pvBS_dx->Fill(dx);
      h_pvBS_dy->Fill(dy);
      if (iv == 0) {
        h_pv0BS_dx->Fill(dx);
        h_pv0BS_dy->Fill(dy);
      }
    }
  }
  h_nValidVertices->Fill(nValidVtx);

  // per-vertex accumulators (indexed by vertex index)
  std::vector<double> vtxSumPt2(nVtx, 0.);
  std::vector<unsigned int> vtxNAssoc(nVtx, 0);

  // -------------------------------------------------------------------
  // Track loop
  // -------------------------------------------------------------------
  for (const auto& trk : tracks) {
    const float pt = trk.tk_pt();
    const float eta = trk.tk_eta();
    const float phi = trk.tk_phi();

    // rebuild a reco::Track from the perigee parameters
    const reco::Track recoTrk = makeRecoTrack(trk);

    // ---- kinematics ----
    h_pt->Fill(pt);
    h_p->Fill(recoTrk.p());
    h_eta->Fill(eta);
    h_phi->Fill(phi);
    h_charge->Fill(trk.tk_charge());
    h2_eta_phi->Fill(eta, phi);

    // ---- fit quality ----
    const double chi2Prob = TMath::Prob(recoTrk.chi2(), recoTrk.ndof());
    const double normchi2 = recoTrk.normalizedChi2();
    h_chi2->Fill(recoTrk.chi2());
    h_ndof->Fill(recoTrk.ndof());
    h_normchi2->Fill(normchi2);
    h_chi2prob->Fill(chi2Prob);

    p_chi2_vs_phi->Fill(phi, recoTrk.chi2());
    p_chi2_vs_eta->Fill(eta, recoTrk.chi2());
    p_normchi2_vs_phi->Fill(phi, normchi2);
    p_normchi2_vs_eta->Fill(eta, normchi2);
    p_normchi2_vs_pt->Fill(pt, normchi2);
    p_normchi2_vs_p->Fill(recoTrk.p(), normchi2);
    p_chi2Prob_vs_phi->Fill(phi, chi2Prob);
    p_chi2Prob_vs_eta->Fill(eta, chi2Prob);

    if (pt > 0.f) {
      const double ptRes = recoTrk.ptError() / pt;
      h_ptErrOverPt->Fill(ptRes);
      p_ptRes_vs_phi->Fill(phi, ptRes);
      p_ptRes_vs_eta->Fill(eta, ptRes);
      p_ptRes_vs_pt->Fill(pt, ptRes);

      // q/pT is proportional to the curvature (no magnetic field needed)
      const double qoverpt = trk.tk_charge() / pt;
      p_qoverpt_vs_phi->Fill(phi, qoverpt);
      p_qoverpt_vs_eta->Fill(eta, qoverpt);
    }

    // ---- perigee parameters and errors ----
    h_qoverp->Fill(trk.tk_qoverp());
    h_lambda->Fill(trk.tk_lambda());
    h_dsz->Fill(trk.tk_dsz());
    h_qoverpErr->Fill(trk.tk_qoverp_Error());
    h_lambdaErr->Fill(trk.tk_lambda_Error());
    h_phiErr->Fill(trk.tk_phi_Error());
    h_dxyErrRaw->Fill(trk.tk_dxy_Error() * cmToUm);
    h_dzErrRaw->Fill(trk.tk_dz_Error() * cmToUm);

    // ---- hits ----
    const int nValidPixelHits = trk.tk_nValidPixelHits();
    const int nTrackerLayers = trk.tk_nTrackerLayersWithMeasurement();
    const int nValidStripHits = trk.tk_nValidStripHits();
    h_nPixelHits->Fill(nValidPixelHits);
    h_nTrackerLayers->Fill(nTrackerLayers);
    h_nStripHits->Fill(nValidStripHits);

    const std::array<double, 3> hitValues = {static_cast<double>(nValidPixelHits),
                                             static_cast<double>(nTrackerLayers),
                                             static_cast<double>(nValidStripHits)};
    for (size_t i = 0; i < hitProfiles_.size(); ++i) {
      hitProfiles_[i].fill(eta, phi, pt, hitValues[i]);
    }

    // ---- reference point ----
    h_vr->Fill(std::hypot(trk.tk_vx(), trk.tk_vy()));
    h_vz->Fill(trk.tk_vz());

    // ---- closure: rebuilt vs stored (both w.r.t. the origin) ----
    h_dxyClosure->Fill((recoTrk.dxy() - trk.tk_dxy()) * cmToUm);
    h_dzClosure->Fill((recoTrk.dz() - trk.tk_dz()) * cmToUm);
    h_dxyErrClosure->Fill((recoTrk.dxyError() - trk.tk_dxy_Error()) * cmToUm);
    h_dzErrClosure->Fill((recoTrk.dzError() - trk.tk_dz_Error()) * cmToUm);

    // ---- impact parameters w.r.t. the beam spot ----
    if (beamSpot) {
      const double dxyBS = recoTrk.dxy(*beamSpot) * cmToUm;
      const double dzBS = recoTrk.dz(beamSpot->position(recoTrk.vz()));
      h_dxyBS->Fill(dxyBS);
      h_dzBS->Fill(dzBS);
      p_dxyBS_vs_phi->Fill(phi, dxyBS);
      p_dxyBS_vs_eta->Fill(eta, dxyBS);
      p_dzBS_vs_phi->Fill(phi, dzBS);
      p_dzBS_vs_eta->Fill(eta, dzBS);
      p_chi2Prob_vs_dxy->Fill(std::abs(dxyBS), chi2Prob);
      p_chi2Prob_vs_dz->Fill(dzBS, chi2Prob);
    }

    // ---- track / vertex association ----
    const int vtxIndex = trk.tk_vtxInd();
    h_vtx_idx->Fill(vtxIndex);

    if (vtxIndex < 0 || static_cast<unsigned int>(vtxIndex) >= nVtx) {
      h_trkAssoc->Fill(0);
      continue;  // no PV: nothing more to do for this track
    }
    h_trkAssoc->Fill(vtxIndex == 0 ? 1 : 2);

    const Run3ScoutingVertex& pv = vertices[vtxIndex];
    if (!pv.isValidVtx())
      continue;

    vtxSumPt2[vtxIndex] += static_cast<double>(pt) * pt;
    vtxNAssoc[vtxIndex]++;

    h_trk_vz_minus_pvz->Fill(trk.tk_vz() - pv.z());

    // ---- impact parameters w.r.t. the associated PV (standard CMSSW definitions) ----
    const reco::Vertex recoVtx = makeRecoVertex(pv);

    const float dxy = recoTrk.dxy(recoVtx.position()) * cmToUm;
    const float dz = recoTrk.dz(recoVtx.position()) * cmToUm;

    // dxy error: track error combined with the vertex covariance projected onto the dxy direction
    const float dxyErr = recoTrk.dxyError(recoVtx.position(), recoVtx.covariance()) * cmToUm;
    // dz error: no projected version available in TrackBase, add the vertex z error in quadrature
    const float dzErr = std::sqrt(recoTrk.dzError() * recoTrk.dzError() + recoVtx.zError() * recoVtx.zError()) * cmToUm;

    h_dxy->Fill(dxy);
    h_dz->Fill(dz);

    const std::array<double, 2> ipValues = {dxy, dz};
    for (size_t i = 0; i < impactParameterProfiles_.size(); ++i) {
      impactParameterProfiles_[i].fill(eta, phi, pt, ipValues[i]);
    }

    if (pt < 1.f)
      continue;

    dxy_pt1.fill(eta, phi, pt, dxy, dxyErr);
    dz_pt1.fill(eta, phi, pt, dz, dzErr);

    if (pt < 10.f)
      continue;

    dxy_pt10.fill(eta, phi, pt, dxy, dxyErr);
    dz_pt10.fill(eta, phi, pt, dz, dzErr);
  }

  // -------------------------------------------------------------------
  // Per-vertex quantities derived from the association
  // -------------------------------------------------------------------
  for (unsigned int iv = 0; iv < nVtx; ++iv) {
    if (!vertices[iv].isValidVtx())
      continue;
    h2_vtx_nAssoc_vs_nTracks->Fill(vertices[iv].tracksSize(), vtxNAssoc[iv]);
    if (vtxNAssoc[iv] > 0)
      h_vtx_sumPt2->Fill(vtxSumPt2[iv]);
  }
}

// helper: build reco::Track
reco::Track ScoutingTrackMonitor::makeRecoTrack(const Run3ScoutingTrack& sTrack) const {
  const reco::Track::Point v(sTrack.tk_vx(), sTrack.tk_vy(), sTrack.tk_vz());
  const reco::Track::Vector p(math::RhoEtaPhiVector(sTrack.tk_pt(), sTrack.tk_eta(), sTrack.tk_phi()));

  // perigee covariance, ordering (qoverp, lambda, phi, dxy, dsz) as in reco::TrackBase
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

void ScoutingTrackMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracks", edm::InputTag("hltScoutingTrackPacker"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"));
  desc.add<edm::InputTag>("beamSpotLabel", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<std::string>("topFolderName", "HLT/ScoutingOffline/Tracks");
  desc.add<double>("Xpos", 0.1);
  desc.add<double>("Ypos", -0.2);
  desc.add<int>("DxyBin", 100);
  desc.add<double>("DxyMin", -5000.0);
  desc.add<double>("DxyMax", 5000.0);
  desc.add<int>("DzBin", 100);
  desc.add<double>("DzMin", -2000.0);
  desc.add<double>("DzMax", 2000.0);
  desc.add<int>("PhiBin", 32);
  desc.add<double>("PhiMin", -std::numbers::pi);
  desc.add<double>("PhiMax", std::numbers::pi);
  desc.add<int>("EtaBin", 26);
  desc.add<double>("EtaMin", -3.0);
  desc.add<double>("EtaMax", 3.0);
  desc.add<int>("PtBin", 49);
  desc.add<double>("PtMin", 1.);
  desc.add<double>("PtMax", 50.);
  desc.add<int>("PhiBin2D", 12);
  desc.add<int>("EtaBin2D", 8);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(ScoutingTrackMonitor);
