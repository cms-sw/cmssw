// system includes
#include <cmath>
#include <vector>
#include <numbers>
#include <fmt/format.h>
#include <boost/range/adaptor/indexed.hpp>

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

namespace sctTrackMonitor {
  // same logic used for the MTV:
  // cf https://github.com/cms-sw/cmssw/blob/master/Validation/RecoTrack/src/MTVHistoProducerAlgoForTracker.cc
  typedef dqm::reco::DQMStore DQMStore;

  inline void setBinLog(TAxis* axis) {
    int bins = axis->GetNbins();
    float from = axis->GetXmin();
    float to = axis->GetXmax();
    float width = (to - from) / bins;
    std::vector<float> new_bins(bins + 1, 0);
    for (int i = 0; i <= bins; i++) {
      new_bins[i] = TMath::Power(10, from + i * width);
    }
    axis->Set(bins, new_bins.data());
  }

  inline void setBinLogX(TH1* h) {
    TAxis* axis = h->GetXaxis();
    setBinLog(axis);
  }
  inline void setBinLogY(TH1* h) {
    TAxis* axis = h->GetYaxis();
    setBinLog(axis);
  }

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

  struct IPMonitoring {
    std::string varname_;
    float pTcut_;
    dqm::reco::MonitorElement *IP_, *IPErr_, *IPPull_;
    dqm::reco::MonitorElement *IPVsPhi_, *IPVsEta_, *IPVsPt_;
    dqm::reco::MonitorElement *IPErrVsPhi_, *IPErrVsEta_, *IPErrVsPt_;
    dqm::reco::MonitorElement *IPVsEtaVsPhi_, *IPErrVsEtaVsPhi_;

    void bookIPMonitor(DQMStore::IBooker&, const edm::ParameterSet&);

  private:
    int PhiBin_, EtaBin_, PtBin_;
    double PhiMin_, PhiMax_, EtaMin_, EtaMax_, PtMin_, PtMax_;
  };

protected:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  static inline std::pair<float, float> trk_vtx_offSet(const Run3ScoutingTrack& tk, const Run3ScoutingVertex& vtx) {
    const auto pt = tk.tk_pt();
    const auto phi = tk.tk_phi();
    const auto eta = tk.tk_eta();

    const auto px = pt * std::cos(phi);
    const auto py = pt * std::sin(phi);
    const auto pz = pt * std::sinh(eta);
    const auto pt2 = pt * pt;

    const auto dx = tk.tk_vx() - vtx.x();
    const auto dy = tk.tk_vy() - vtx.y();
    const auto dz = tk.tk_vz() - vtx.z();

    const auto tk_dxyPV = (-dx * py + dy * px) / pt;
    const auto tk_dzPV = dz - (dx * px + dy * py) * pz / pt2;

    return {tk_dxyPV, tk_dzPV};
  }

  // configuration
  const edm::ParameterSet conf_;

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

  // IP monitoring structs
  IPMonitoring dxy_pt1;
  IPMonitoring dxy_pt10;

  IPMonitoring dz_pt1;
  IPMonitoring dz_pt10;

  // profiles
  std::vector<MonitorElement*> vTrackProfiles_;

  // helpers
  reco::Track makeRecoTrack(const Run3ScoutingTrack& sTrack) const;
  reco::Vertex makeRecoVertex(const Run3ScoutingVertex& sVertex) const;
  std::pair<unsigned int, const Run3ScoutingVertex*> findClosestScoutingVertex(
      const reco::Track* track, const std::vector<Run3ScoutingVertex>& vertices);

  template <class OBJECT_TYPE>
  int index(const std::vector<OBJECT_TYPE*>& vec, const TString& name) {
    for (const auto& iter : vec | boost::adaptors::indexed(0)) {
      if (iter.value() && iter.value()->getName() == name) {
        return iter.index();
      }
    }
    edm::LogError("ScoutingTrackMonitor") << "@SUB=ScoutingTrackMonitor::index"
                                          << " could not find " << name;
    return -1;
  }
};

// constructor
ScoutingTrackMonitor::ScoutingTrackMonitor(const edm::ParameterSet& iConfig)
    : conf_(iConfig),
      tracksToken_{consumes<std::vector<Run3ScoutingTrack>>(iConfig.getParameter<edm::InputTag>("tracks"))},
      verticesToken_{consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("vertices"))},
      topFolderName_{iConfig.getParameter<std::string>("topFolderName")} {}

// histogram booking
void ScoutingTrackMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(topFolderName_);

  h_dxy = ibooker.book1D("dxy", "d_{xy};d_{xy} [#mum];Tracks", 100, -0.15 * cmToUm, 0.15 * cmToUm);
  h_dz = ibooker.book1D("dz", "d_{z};d_{z} [#mum];Tracks", 100, -0.35 * cmToUm, 0.35 * cmToUm);

  h_vtx_idx = ibooker.book1DD("vertexIndex", "tracks Vertex Index;Vertex index;Tracks", 17, -1.5, 15.5);

  p_dxy_eta = ibooker.bookProfile(
      "dxy_vs_eta", "d_{xy} vs #eta;#eta;#LTd_{xy}#GT [#mum]", 50, -3.0, 3.0, -0.01 * cmToUm, 0.01 * cmToUm, "");
  p_dxy_phi = ibooker.bookProfile("dxy_vs_phi",
                                  "d_{xy} vs #phi;#phi [rad];#LTd_{xy}#GT [#mum]",
                                  50,
                                  -std::numbers::pi,
                                  std::numbers::pi,
                                  -0.15 * cmToUm,
                                  0.15 * cmToUm,
                                  "");
  p_dz_eta = ibooker.bookProfile(
      "dz_vs_eta", "d_{z} vs #eta;#eta;#LTd_{z}#GT [#mum]", 50, -3.0, 3.0, -0.05 * cmToUm, 0.05 * cmToUm, "");
  p_dz_phi = ibooker.bookProfile("dz_vs_phi",
                                 "d_{z} vs #phi;#phi [rad];#LTd_{z}#GT [#mum]",
                                 50,
                                 -std::numbers::pi,
                                 std::numbers::pi,
                                 -0.35 * cmToUm,
                                 0.35 * cmToUm,
                                 "");

  // 2D eta-phi occupancy histograms
  h2_eta_phi = ibooker.book2I(
      "eta_vs_phi", "Track occupancy;#eta;#phi [rad]", 50, -3.0, 3.0, 50, -std::numbers::pi, std::numbers::pi);
  h2_eta_phi->setOption("colz");

  // 2D eta-phi profiles
  p2_dxy_eta_phi = ibooker.bookProfile2D("dxy_vs_eta_phi",
                                         "d_{xy} vs #eta-#phi;#eta;#phi [rad];#LTd_{xy}#GT [#mum]",
                                         50,
                                         -3.0,
                                         3.0,
                                         50,
                                         -std::numbers::pi,
                                         std::numbers::pi,
                                         -0.15 * cmToUm,
                                         0.15 * cmToUm,
                                         "");
  p2_dxy_eta_phi->setOption("colz");

  p2_dz_eta_phi = ibooker.bookProfile2D("dz_vs_eta_phi",
                                        "d_{z} vs #eta-#phi;#eta;#phi [rad];#LTd_{z}#GT [#mum]",
                                        50,
                                        -3.0,
                                        3.0,
                                        50,
                                        -std::numbers::pi,
                                        std::numbers::pi,
                                        -0.35 * cmToUm,
                                        0.35 * cmToUm,
                                        "");
  p2_dz_eta_phi->setOption("colz");

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
  p2_nValidPixelHits_eta_phi->setOption("colz");

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
  p2_nTrackerLayersWithMeasurement_eta_phi->setOption("colz");

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
  p2_nValidStripHits_eta_phi->setOption("colz");

  // intialize the profiles
  double xBins[19] = {0., 0.15, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 7., 10., 15., 25., 40., 100., 200.};
  vTrackProfiles_.push_back(ibooker.bookProfile("p_d0_vs_phi",
                                                "Transverse Impact Parameter vs. #phi;#phi_{Track};#LT d_{0} #GT [cm]",
                                                100,
                                                -std::numbers::pi,
                                                std::numbers::pi,
                                                -0.15 * cmToUm,
                                                0.15 * cmToUm,
                                                ""));
  vTrackProfiles_.push_back(
      ibooker.bookProfile("p_dz_vs_phi",
                          "Longitudinal Impact Parameter vs. #phi;#phi_{Track};#LT d_{z} #GT [cm]",
                          100,
                          -std::numbers::pi,
                          std::numbers::pi,
                          -0.35 * cmToUm,
                          0.35 * cmToUm,
                          ""));
  vTrackProfiles_.push_back(ibooker.bookProfile("p_d0_vs_eta",
                                                "Transverse Impact Parameter vs. #eta;#eta_{Track};#LT d_{0} #GT [cm]",
                                                100,
                                                -3.,
                                                3.,
                                                -0.15 * cmToUm,
                                                0.15 * cmToUm,
                                                ""));
  vTrackProfiles_.push_back(
      ibooker.bookProfile("p_dz_vs_eta",
                          "Longitudinal Impact Parameter vs. #eta;#eta_{Track};#LT d_{z} #GT [cm]",
                          100,
                          -3.,
                          3.,
                          -0.35 * cmToUm,
                          0.35 * cmToUm,
                          ""));
  vTrackProfiles_.push_back(ibooker.bookProfile("p_chi2_vs_phi",
                                                "#chi^{2} vs. #phi;#phi_{Track};#LT #chi^{2} #GT",
                                                100,
                                                -std::numbers::pi,
                                                std::numbers::pi,
                                                0,
                                                100,
                                                ""));
  vTrackProfiles_.push_back(
      ibooker.bookProfile("p_chi2Prob_vs_phi",
                          "#chi^{2} probablility vs. #phi;#phi_{Track};#LT #chi^{2} probability#GT",
                          100,
                          -std::numbers::pi,
                          std::numbers::pi,
                          0.,
                          1.,
                          ""));
  vTrackProfiles_.push_back(
      ibooker.bookProfile("p_chi2Prob_vs_d0",
                          "#chi^{2} probablility vs. |d_{0}|;|d_{0}|[cm];#LT #chi^{2} probability#GT",
                          100,
                          0,
                          80,
                          0.,
                          1.,
                          ""));
  vTrackProfiles_.push_back(ibooker.bookProfile("p_chi2Prob_vs_dz",
                                                "#chi^{2} probablility vs. dz;d_{z} [cm];#LT #chi^{2} probability#GT",
                                                100,
                                                -30,
                                                30,
                                                0.,
                                                1.,
                                                ""));
  vTrackProfiles_.push_back(ibooker.bookProfile("p_normchi2_vs_phi",
                                                "#chi^{2}/ndof vs. #phi;#phi_{Track};#LT #chi^{2}/ndof #GT",
                                                100,
                                                -std::numbers::pi,
                                                std::numbers::pi,
                                                0.,
                                                5.,
                                                ""));
  vTrackProfiles_.push_back(ibooker.bookProfile(
      "p_chi2_vs_eta", "#chi^{2} vs. #eta;#eta_{Track};#LT #chi^{2} #GT", 100, -3., 3., 0., 100., ""));
  vTrackProfiles_.push_back(ibooker.bookProfile("p_normchi2_vs_pt",
                                                "norm #chi^{2} vs. p_{T}_{Track}; p_{T}_{Track};#LT #chi^{2}/ndof #GT",
                                                18,
                                                xBins,
                                                0.,
                                                5.,
                                                ""));
  vTrackProfiles_.push_back(ibooker.bookProfile(
      "p_normchi2_vs_p", "#chi^{2}/ndof vs. p_{Track};p_{Track};#LT #chi^{2}/ndof #GT", 18, xBins, 0., 5., ""));
  vTrackProfiles_.push_back(
      ibooker.bookProfile("p_chi2Prob_vs_eta",
                          "#chi^{2} probability vs. #eta;#eta_{Track};#LT #chi^{2} probability #GT",
                          100,
                          -3.,
                          3.,
                          0.,
                          1.,
                          ""));
  vTrackProfiles_.push_back(ibooker.bookProfile(
      "p_normchi2_vs_eta", "#chi^{2}/ndof vs. #eta;#eta_{Track};#LT #chi^{2}/ndof #GT", 100, -3., 3., 0., 5, ""));
  vTrackProfiles_.push_back(ibooker.bookProfile(
      "p_kappa_vs_phi", "#kappa vs. #phi;#phi_{Track};#kappa", 100, -std::numbers::pi, std::numbers::pi, -5., 5., ""));
  vTrackProfiles_.push_back(
      ibooker.bookProfile("p_kappa_vs_eta", "#kappa vs. #eta;#eta_{Track};#kappa", 100, -3., 3., -5., 5., ""));
  vTrackProfiles_.push_back(
      ibooker.bookProfile("p_ptResolution_vs_phi",
                          "#delta_{p_{T}}/p_{T}^{track};#phi^{track};#delta_{p_{T}}/p_{T}^{track}",
                          100,
                          -std::numbers::pi,
                          std::numbers::pi,
                          0.,
                          1.,
                          ""));
  vTrackProfiles_.push_back(
      ibooker.bookProfile("p_ptResolution_vs_eta",
                          "#delta_{p_{T}}/p_{T}^{track};#eta^{track};#delta_{p_{T}}/p_{T}^{track}",
                          100,
                          -3.,
                          3.,
                          0.,
                          1.,
                          ""));

  // initialize and book the monitors;
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
}

void ScoutingTrackMonitor::IPMonitoring::bookIPMonitor(DQMStore::IBooker& iBooker, const edm::ParameterSet& config) {
  int VarBin = config.getParameter<int>(fmt::format("D{}Bin", varname_));
  double VarMin = config.getParameter<double>(fmt::format("D{}Min", varname_));
  double VarMax = config.getParameter<double>(fmt::format("D{}Max", varname_));

  PhiBin_ = config.getParameter<int>("PhiBin");
  PhiMin_ = config.getParameter<double>("PhiMin");
  PhiMax_ = config.getParameter<double>("PhiMax");
  int PhiBin2D = config.getParameter<int>("PhiBin2D");

  EtaBin_ = config.getParameter<int>("EtaBin");
  EtaMin_ = config.getParameter<double>("EtaMin");
  EtaMax_ = config.getParameter<double>("EtaMax");
  int EtaBin2D = config.getParameter<int>("EtaBin2D");

  PtBin_ = config.getParameter<int>("PtBin");
  PtMin_ = config.getParameter<double>("PtMin") * pTcut_;
  PtMax_ = config.getParameter<double>("PtMax") * pTcut_;

  // 1D variables

  IP_ = iBooker.book1D(fmt::format("d{}_pt{}", varname_, pTcut_),
                       fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_),
                       VarBin,
                       VarMin,
                       VarMax);

  IPErr_ = iBooker.book1D(fmt::format("d{}Err_pt{}", varname_, pTcut_),
                          fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_),
                          100,
                          0.,
                          (varname_.find("xy") != std::string::npos) ? 2000. : 10000.);

  IPPull_ = iBooker.book1D(
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
  IPVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #phi", 1);
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
  IPVsEta_->setAxisTitle("PV track (p_{T} > 1 GeV) #eta", 1);
  IPVsEta_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_), 2);

  IPVsPt_ = sctTrackMonitor::makeProfileIfLog(
      iBooker,
      true,  /* x-axis */
      false, /* y-axis */
      fmt::format("d{}VsPt_pt{}", varname_, pTcut_).c_str(),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track p_{{T}}", pTcut_, varname_).c_str(),
      PtBin_,
      log10(PtMin_),
      log10(PtMax_),
      VarMin,
      VarMax,
      "");
  IPVsPt_->setAxisTitle("PV track (p_{T} > 1 GeV) p_{T} [GeV]", 1);
  IPVsPt_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_), 2);

  // IP error profiles

  IPErrVsPhi_ =
      iBooker.bookProfile(fmt::format("d{}ErrVsPhi_pt{}", varname_, pTcut_),
                          fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #phi", pTcut_, varname_),
                          PhiBin_,
                          PhiMin_,
                          PhiMax_,
                          VarBin,
                          0.,
                          (varname_.find("xy") != std::string::npos) ? 100. : 200.,
                          "");
  IPErrVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #phi", 1);
  IPErrVsPhi_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_), 2);

  IPErrVsEta_ =
      iBooker.bookProfile(fmt::format("d{}ErrVsEta_pt{}", varname_, pTcut_),
                          fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #eta", pTcut_, varname_),
                          EtaBin_,
                          EtaMin_,
                          EtaMax_,
                          VarBin,
                          0.,
                          (varname_.find("xy") != std::string::npos) ? 100. : 200.,
                          "");
  IPErrVsEta_->setAxisTitle("PV track (p_{T} > 1 GeV) #eta", 1);
  IPErrVsEta_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_), 2);

  IPErrVsPt_ = sctTrackMonitor::makeProfileIfLog(
      iBooker,
      true,  /* x-axis */
      false, /* y-axis */
      fmt::format("d{}ErrVsPt_pt{}", varname_, pTcut_).c_str(),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track p_{{T}}", pTcut_, varname_).c_str(),
      PtBin_,
      log10(PtMin_),
      log10(PtMax_),
      VarMin,
      VarMax,
      "");
  IPErrVsPt_->setAxisTitle("PV track (p_{T} > 1 GeV) p_{T} [GeV]", 1);
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
  IPVsEtaVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #eta", 1);
  IPVsEtaVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #phi", 2);
  IPVsEtaVsPhi_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_), 3);

  IPErrVsEtaVsPhi_ = iBooker.bookProfile2D(
      fmt::format("d{}ErrVsEtaVsPhi_pt{}", varname_, pTcut_),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #eta VS track #phi", pTcut_, varname_),
      EtaBin2D,
      EtaMin_,
      EtaMax_,
      PhiBin2D,
      PhiMin_,
      PhiMax_,
      VarBin,
      0.,
      (varname_.find("xy") != std::string::npos) ? 100. : 200.,
      "");
  IPErrVsEtaVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #eta", 1);
  IPErrVsEtaVsPhi_->setAxisTitle("PV track (p_{T} > 1 GeV) #phi", 2);
  IPErrVsEtaVsPhi_->setAxisTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_),
                                 3);
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

    // // initialize the impact parameters to large values
    // std::pair<float, float> best_offset{9999.f, 99999.f};

    // // loop on all the vertices and find the closest one
    // unsigned int vtxIndex = 999;
    // unsigned int idx = 0;
    // for (const auto& vtx : vertices) {
    //   const auto offset = trk_vtx_offSet(trk, vtx);
    //   if (std::abs(offset.second) < std::abs(best_offset.second)) {
    //     best_offset = offset;
    //     vtxIndex = idx;  // save the index of the best vertex
    //   }
    //   idx++;
    // }

    h_vtx_idx->Fill(vtxIndex);

    const float eta = trk.tk_eta();
    const float phi = trk.tk_phi();
    const float pt = trk.tk_pt();

    // --- fill 2D eta-phi occupancy histograms ---
    h2_eta_phi->Fill(eta, phi);
    p2_nValidPixelHits_eta_phi->Fill(eta, phi, trk.tk_nValidPixelHits());
    p2_nTrackerLayersWithMeasurement_eta_phi->Fill(eta, phi, trk.tk_nTrackerLayersWithMeasurement());
    p2_nValidStripHits_eta_phi->Fill(eta, phi, trk.tk_nValidStripHits());

    // --- build reco vertex ---
    reco::Vertex recoVtx = makeRecoVertex(*closestVtx);

    // --- impact parameters (standard CMSSW definitions) ---
    float dxy = recoTrk.dxy(recoVtx.position()) * cmToUm;
    float dz = recoTrk.dz(recoVtx.position()) * cmToUm;
    float dxyErr = recoTrk.dxyError() * cmToUm;
    float dzErr = recoTrk.dzError() * cmToUm;

    //float dxy = best_offset.first;
    //float dz = best_offset.second;

    // --- fill histograms ---
    h_dxy->Fill(dxy);
    h_dz->Fill(dz);

    p_dxy_eta->Fill(eta, dxy);
    p_dxy_phi->Fill(phi, dxy);

    p_dz_eta->Fill(eta, dz);
    p_dz_phi->Fill(phi, dz);

    // --- fill 2D eta-phi profiles ---
    p2_dxy_eta_phi->Fill(eta, phi, dxy);
    p2_dz_eta_phi->Fill(eta, phi, dz);

    // Fill track profiles
    double chi2Prob = TMath::Prob(recoTrk.chi2(), recoTrk.ndof());
    double normchi2 = recoTrk.normalizedChi2();
    double kappa = trk.tk_qoverp();

    //GlobalPoint gPoint(recoTrk.vx(), recoTrk.vy(), recoTrk.vz());
    //double theLocalMagFieldInInverseGeV = magneticField_->inInverseGeV(gPoint).z();
    //double kappa = -recoTrk.charge() * theLocalMagFieldInInverseGeV / recoTrk.pt();

    static const int d0phiindex = this->index(vTrackProfiles_, "p_d0_vs_phi");
    vTrackProfiles_[d0phiindex]->Fill(recoTrk.phi(), recoTrk.d0());
    static const int dzphiindex = this->index(vTrackProfiles_, "p_dz_vs_phi");
    vTrackProfiles_[dzphiindex]->Fill(recoTrk.phi(), recoTrk.dz());
    static const int d0etaindex = this->index(vTrackProfiles_, "p_d0_vs_eta");
    vTrackProfiles_[d0etaindex]->Fill(recoTrk.eta(), recoTrk.d0());
    static const int dzetaindex = this->index(vTrackProfiles_, "p_dz_vs_eta");
    vTrackProfiles_[dzetaindex]->Fill(recoTrk.eta(), recoTrk.dz());
    static const int chiProbphiindex = this->index(vTrackProfiles_, "p_chi2Prob_vs_phi");
    vTrackProfiles_[chiProbphiindex]->Fill(recoTrk.phi(), chi2Prob);
    static const int chiProbabsd0index = this->index(vTrackProfiles_, "p_chi2Prob_vs_d0");
    vTrackProfiles_[chiProbabsd0index]->Fill(fabs(recoTrk.d0()), chi2Prob);
    static const int chiProbabsdzindex = this->index(vTrackProfiles_, "p_chi2Prob_vs_dz");
    vTrackProfiles_[chiProbabsdzindex]->Fill(recoTrk.dz(), chi2Prob);
    static const int chiphiindex = this->index(vTrackProfiles_, "p_chi2_vs_phi");
    vTrackProfiles_[chiphiindex]->Fill(recoTrk.phi(), recoTrk.chi2());
    static const int normchiphiindex = this->index(vTrackProfiles_, "p_normchi2_vs_phi");
    vTrackProfiles_[normchiphiindex]->Fill(recoTrk.phi(), normchi2);
    static const int chietaindex = this->index(vTrackProfiles_, "p_chi2_vs_eta");
    vTrackProfiles_[chietaindex]->Fill(recoTrk.eta(), recoTrk.chi2());
    static const int normchiptindex = this->index(vTrackProfiles_, "p_normchi2_vs_pt");
    vTrackProfiles_[normchiptindex]->Fill(recoTrk.pt(), normchi2);
    static const int normchipindex = this->index(vTrackProfiles_, "p_normchi2_vs_p");
    vTrackProfiles_[normchipindex]->Fill(recoTrk.p(), normchi2);
    static const int chiProbetaindex = this->index(vTrackProfiles_, "p_chi2Prob_vs_eta");
    vTrackProfiles_[chiProbetaindex]->Fill(recoTrk.eta(), chi2Prob);
    static const int normchietaindex = this->index(vTrackProfiles_, "p_normchi2_vs_eta");
    vTrackProfiles_[normchietaindex]->Fill(recoTrk.eta(), normchi2);
    static const int kappaphiindex = this->index(vTrackProfiles_, "p_kappa_vs_phi");
    vTrackProfiles_[kappaphiindex]->Fill(recoTrk.phi(), kappa);
    static const int kappaetaindex = this->index(vTrackProfiles_, "p_kappa_vs_eta");
    vTrackProfiles_[kappaetaindex]->Fill(recoTrk.eta(), kappa);
    static const int ptResphiindex = this->index(vTrackProfiles_, "p_ptResolution_vs_phi");
    vTrackProfiles_[ptResphiindex]->Fill(recoTrk.phi(), recoTrk.ptError() / recoTrk.pt());
    static const int ptResetaindex = this->index(vTrackProfiles_, "p_ptResolution_vs_eta");
    vTrackProfiles_[ptResetaindex]->Fill(recoTrk.eta(), recoTrk.ptError() / recoTrk.pt());

    if (trk.tk_pt() < 1.)
      continue;

    dxy_pt1.IP_->Fill(dxy);
    dxy_pt1.IPVsPhi_->Fill(phi, dxy);
    dxy_pt1.IPVsEta_->Fill(eta, dxy);
    dxy_pt1.IPVsPt_->Fill(pt, dxy);
    dxy_pt1.IPVsEtaVsPhi_->Fill(eta, phi, dxy);

    dxy_pt1.IPErr_->Fill(dxyErr);
    dxy_pt1.IPPull_->Fill(dxy / dxyErr);
    dxy_pt1.IPErrVsPhi_->Fill(phi, dxyErr);
    dxy_pt1.IPErrVsEta_->Fill(eta, dxyErr);
    dxy_pt1.IPErrVsPt_->Fill(pt, dxyErr);
    dxy_pt1.IPErrVsEtaVsPhi_->Fill(eta, phi, dxyErr);

    // dz pT>1

    dz_pt1.IP_->Fill(dz);
    dz_pt1.IPVsPhi_->Fill(phi, dz);
    dz_pt1.IPVsEta_->Fill(eta, dz);
    dz_pt1.IPVsPt_->Fill(pt, dz);
    dz_pt1.IPVsEtaVsPhi_->Fill(eta, phi, dz);

    dz_pt1.IPErr_->Fill(dzErr);
    dz_pt1.IPPull_->Fill(dz / dzErr);
    dz_pt1.IPErrVsPhi_->Fill(phi, dzErr);
    dz_pt1.IPErrVsEta_->Fill(eta, dzErr);
    dz_pt1.IPErrVsPt_->Fill(pt, dzErr);
    dz_pt1.IPErrVsEtaVsPhi_->Fill(eta, phi, dzErr);

    if (pt < 10.)
      continue;

    // dxy pT>10
    dxy_pt10.IP_->Fill(dxy);
    dxy_pt10.IPVsPhi_->Fill(phi, dxy);
    dxy_pt10.IPVsEta_->Fill(eta, dxy);
    dxy_pt10.IPVsPt_->Fill(pt, dxy);
    dxy_pt10.IPVsEtaVsPhi_->Fill(eta, phi, dxy);

    dxy_pt10.IPErr_->Fill(dxyErr);
    dxy_pt10.IPPull_->Fill(dxy / dxyErr);
    dxy_pt10.IPErrVsPhi_->Fill(phi, dxyErr);
    dxy_pt10.IPErrVsEta_->Fill(eta, dxyErr);
    dxy_pt10.IPErrVsPt_->Fill(pt, dxyErr);
    dxy_pt10.IPErrVsEtaVsPhi_->Fill(eta, phi, dxyErr);

    // dxz pT>10
    dz_pt10.IP_->Fill(dz);
    dz_pt10.IPVsPhi_->Fill(phi, dz);
    dz_pt10.IPVsEta_->Fill(eta, dz);
    dz_pt10.IPVsPt_->Fill(pt, dz);
    dz_pt10.IPVsEtaVsPhi_->Fill(eta, phi, dz);

    dz_pt10.IPErr_->Fill(dzErr);
    dz_pt10.IPPull_->Fill(dz / dzErr);
    dz_pt10.IPErrVsPhi_->Fill(phi, dzErr);
    dz_pt10.IPErrVsEta_->Fill(eta, dzErr);
    dz_pt10.IPErrVsPt_->Fill(pt, dzErr);
    dz_pt10.IPErrVsEtaVsPhi_->Fill(eta, phi, dzErr);
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
  desc.add<double>("EtaMin", 2.5);
  desc.add<double>("EtaMax", -2.5);
  desc.add<int>("PtBin", 49);
  desc.add<double>("PtMin", 1.);
  desc.add<double>("PtMax", 50.);
  desc.add<int>("PhiBin2D", 12);
  desc.add<int>("EtaBin2D", 8);
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ScoutingTrackMonitor);
