// -*- C++ -*-
//
// Package:    Alignment/GeneralPurposeVertexAnalyzer
// Class:      GeneralPurposeVertexAnalyzer
//
/**\class GeneralPurposeVertexAnalyzer GeneralPurposeVertexAnalyzer.cc Alignment/GeneralPurposeVertexAnalyzer/plugins/GeneralPurposeVertexAnalyzer.cc
   Description: monitor vertex properties for alignment purposes, largely copied from DQMOffline/RecoB/plugins/PrimaryVertexMonitor.cc

*/
//
// Original Author:  Marco Musich
//         Created:  Thu, 13 Apr 2023 14:16:43 GMT
//
//

// ROOT includes files
#include "TMath.h"
#include "TFile.h"
#include "TH1I.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TProfile2D.h"

// system include files
#include <memory>
#include <fmt/format.h>
#include <fmt/printf.h>

// user include files
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/isFinite.h"

//
// class declaration
//

using reco::TrackCollection;

namespace gpVertexAnalyzer {
  void setBinLog(TAxis *axis) {
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

  void setBinLogX(TH1 *h) {
    TAxis *axis = h->GetXaxis();
    setBinLog(axis);
  }

  void setBinLogY(TH1 *h) {
    TAxis *axis = h->GetYaxis();
    setBinLog(axis);
  }

  template <typename... Args>
  TProfile *makeProfileIfLog(const edm::Service<TFileService> &fs, bool logx, bool logy, Args &&...args) {
    auto prof = fs->make<TProfile>(std::forward<Args>(args)...);
    if (logx)
      setBinLogX(prof);
    if (logy)
      setBinLogY(prof);
    return prof;
  }

  template <typename... Args>
  TH1D *makeTH1IfLog(const edm::Service<TFileService> &fs, bool logx, bool logy, Args &&...args) {
    auto h1 = fs->make<TH1D>(std::forward<Args>(args)...);
    if (logx)
      setBinLogX(h1);
    if (logy)
      setBinLogY(h1);
    return h1;
  }
}  // namespace gpVertexAnalyzer

class GeneralPurposeVertexAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit GeneralPurposeVertexAnalyzer(const edm::ParameterSet &);
  ~GeneralPurposeVertexAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void pvTracksPlots(const reco::Vertex &v);
  void vertexPlots(const reco::Vertex &v, const reco::BeamSpot &beamSpot, int i);
  template <typename T, typename... Args>
  T *book(const Args &...args) const;
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  // ----------member data ---------------------------
  edm::Service<TFileService> fs_;

  struct IPMonitoring {
    std::string varname_;
    float pTcut_;
    TH1D *IP_, *IPErr_, *IPPull_;
    TProfile *IPVsPhi_, *IPVsEta_, *IPVsPt_;
    TProfile *IPErrVsPhi_, *IPErrVsEta_, *IPErrVsPt_;
    TProfile2D *IPVsEtaVsPhi_, *IPErrVsEtaVsPhi_;

    void bookIPMonitor(const edm::ParameterSet &, const edm::Service<TFileService> fs);
  };

  const edm::ParameterSet conf_;
  const int ndof_;
  bool errorPrinted_;

  const edm::InputTag vertexInputTag_, beamSpotInputTag_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  using VertexScore = edm::ValueMap<float>;
  const edm::EDGetTokenT<VertexScore> scoreToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;

  static constexpr int cmToUm = 10000;

  const double vposx_;
  const double vposy_;
  const int tkNoBin_;
  const double tkNoMin_;
  const double tkNoMax_;

  const int dxyBin_;
  const double dxyMin_;
  const double dxyMax_;

  const int dzBin_;
  const double dzMin_;
  const double dzMax_;

  const int phiBin_;
  const int phiBin2D_;
  const double phiMin_;
  const double phiMax_;

  const int etaBin_;
  const int etaBin2D_;
  const double etaMin_;
  const double etaMax_;

  // the histos
  TH1I *nbvtx, *nbgvtx;
  TH1D *nbtksinvtx[2], *trksWeight[2], *score[2];
  TH1D *tt[2];
  TH1D *xrec[2], *yrec[2], *zrec[2], *xDiff[2], *yDiff[2], *xerr[2], *yerr[2], *zerr[2];
  TH2D *xerrVsTrks[2], *yerrVsTrks[2], *zerrVsTrks[2];
  TH1D *ntracksVsZ[2];
  TH1D *vtxchi2[2], *vtxndf[2], *vtxprob[2], *nans[2];
  TH1D *type[2];
  TH1D *bsX, *bsY, *bsZ, *bsSigmaZ, *bsDxdz, *bsDydz, *bsBeamWidthX, *bsBeamWidthY, *bsType;

  TH1D *trackpt, *sumpt, *ntracks, *weight, *chi2ndf, *chi2prob;
  TH1D *dxy2;
  TH1D *phi_pt1, *eta_pt1;
  TH1D *phi_pt10, *eta_pt10;

  // IP monitoring structs
  IPMonitoring dxy_pt1;
  IPMonitoring dxy_pt10;

  IPMonitoring dz_pt1;
  IPMonitoring dz_pt10;
};

void GeneralPurposeVertexAnalyzer::IPMonitoring::bookIPMonitor(const edm::ParameterSet &config,
                                                               const edm::Service<TFileService> fs) {
  int VarBin = config.getParameter<int>(fmt::format("D{}Bin", varname_));
  double VarMin = config.getParameter<double>(fmt::format("D{}Min", varname_));
  double VarMax = config.getParameter<double>(fmt::format("D{}Max", varname_));

  int PhiBin = config.getParameter<int>("PhiBin");
  int PhiBin2D = config.getParameter<int>("PhiBin2D");
  double PhiMin = config.getParameter<double>("PhiMin");
  double PhiMax = config.getParameter<double>("PhiMax");

  int EtaBin = config.getParameter<int>("EtaBin");
  int EtaBin2D = config.getParameter<int>("EtaBin2D");
  double EtaMin = config.getParameter<double>("EtaMin");
  double EtaMax = config.getParameter<double>("EtaMax");

  int PtBin = config.getParameter<int>("PtBin");
  double PtMin = config.getParameter<double>("PtMin") * pTcut_;
  double PtMax = config.getParameter<double>("PtMax") * pTcut_;

  IP_ = fs->make<TH1D>(fmt::format("d{}_pt{}", varname_, pTcut_).c_str(),
                       fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_).c_str(),
                       VarBin,
                       VarMin,
                       VarMax);

  IPErr_ = fs->make<TH1D>(fmt::format("d{}Err_pt{}", varname_, pTcut_).c_str(),
                          fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_).c_str(),
                          100,
                          0.,
                          (varname_.find("xy") != std::string::npos) ? 2000. : 10000.);

  IPPull_ = fs->make<TH1D>(
      fmt::format("d{}Pull_pt{}", varname_, pTcut_).c_str(),
      fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}}/#sigma_{{d_{{{}}}}}", pTcut_, varname_, varname_).c_str(),
      100,
      -5.,
      5.);

  IPVsPhi_ =
      fs->make<TProfile>(fmt::format("d{}VsPhi_pt{}", varname_, pTcut_).c_str(),
                         fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track #phi", pTcut_, varname_).c_str(),
                         PhiBin,
                         PhiMin,
                         PhiMax,
                         //VarBin,
                         VarMin,
                         VarMax);
  IPVsPhi_->SetXTitle("PV track (p_{T} > 1 GeV) #phi");
  IPVsPhi_->SetYTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_).c_str());

  IPVsEta_ =
      fs->make<TProfile>(fmt::format("d{}VsEta_pt{}", varname_, pTcut_).c_str(),
                         fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track #eta", pTcut_, varname_).c_str(),
                         EtaBin,
                         EtaMin,
                         EtaMax,
                         //VarBin,
                         VarMin,
                         VarMax);
  IPVsEta_->SetXTitle("PV track (p_{T} > 1 GeV) #eta");
  IPVsEta_->SetYTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_).c_str());

  IPVsPt_ = gpVertexAnalyzer::makeProfileIfLog(
      fs,
      true,  /* x-axis */
      false, /* y-axis */
      fmt::format("d{}VsPt_pt{}", varname_, pTcut_).c_str(),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track p_{{T}}", pTcut_, varname_).c_str(),
      PtBin,
      log10(PtMin),
      log10(PtMax),
      VarMin,
      VarMax,
      "");
  IPVsPt_->SetXTitle("PV track (p_{T} > 1 GeV) p_{T} [GeV]");
  IPVsPt_->SetYTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_).c_str());

  IPErrVsPhi_ =
      fs->make<TProfile>(fmt::format("d{}ErrVsPhi_pt{}", varname_, pTcut_).c_str(),
                         fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #phi", pTcut_, varname_).c_str(),
                         PhiBin,
                         PhiMin,
                         PhiMax,
                         //VarBin,
                         0.,
                         (varname_.find("xy") != std::string::npos) ? 100. : 200.);
  IPErrVsPhi_->SetXTitle("PV track (p_{T} > 1 GeV) #phi");
  IPErrVsPhi_->SetYTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_).c_str());

  IPErrVsEta_ =
      fs->make<TProfile>(fmt::format("d{}ErrVsEta_pt{}", varname_, pTcut_).c_str(),
                         fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #eta", pTcut_, varname_).c_str(),
                         EtaBin,
                         EtaMin,
                         EtaMax,
                         //VarBin,
                         0.,
                         (varname_.find("xy") != std::string::npos) ? 100. : 200.);
  IPErrVsEta_->SetXTitle("PV track (p_{T} > 1 GeV) #eta");
  IPErrVsEta_->SetYTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_).c_str());

  IPErrVsPt_ = gpVertexAnalyzer::makeProfileIfLog(
      fs,
      true,  /* x-axis */
      false, /* y-axis */
      fmt::format("d{}ErrVsPt_pt{}", varname_, pTcut_).c_str(),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track p_{{T}}", pTcut_, varname_).c_str(),
      PtBin,
      log10(PtMin),
      log10(PtMax),
      VarMin,
      VarMax,
      "");
  IPErrVsPt_->SetXTitle("PV track (p_{T} > 1 GeV) p_{T} [GeV]");
  IPErrVsPt_->SetYTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_).c_str());

  IPVsEtaVsPhi_ = fs->make<TProfile2D>(
      fmt::format("d{}VsEtaVsPhi_pt{}", varname_, pTcut_).c_str(),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} VS track #eta VS track #phi", pTcut_, varname_).c_str(),
      EtaBin2D,
      EtaMin,
      EtaMax,
      PhiBin2D,
      PhiMin,
      PhiMax,
      //VarBin,
      VarMin,
      VarMax);
  IPVsEtaVsPhi_->SetXTitle("PV track (p_{T} > 1 GeV) #eta");
  IPVsEtaVsPhi_->SetYTitle("PV track (p_{T} > 1 GeV) #phi");
  IPVsEtaVsPhi_->SetZTitle(fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} (#mum)", pTcut_, varname_).c_str());

  IPErrVsEtaVsPhi_ = fs->make<TProfile2D>(
      fmt::format("d{}ErrVsEtaVsPhi_pt{}", varname_, pTcut_).c_str(),
      fmt::format("PV tracks (p_{{T}} > {}) d_{{{}}} error VS track #eta VS track #phi", pTcut_, varname_).c_str(),
      EtaBin2D,
      EtaMin,
      EtaMax,
      PhiBin2D,
      PhiMin,
      PhiMax,
      // VarBin,
      0.,
      (varname_.find("xy") != std::string::npos) ? 100. : 200.);
  IPErrVsEtaVsPhi_->SetXTitle("PV track (p_{T} > 1 GeV) #eta");
  IPErrVsEtaVsPhi_->SetYTitle("PV track (p_{T} > 1 GeV) #phi");
  IPErrVsEtaVsPhi_->SetZTitle(
      fmt::format("PV tracks (p_{{T}} > {} GeV) d_{{{}}} error (#mum)", pTcut_, varname_).c_str());
}

//
// constructors and destructor
//
GeneralPurposeVertexAnalyzer::GeneralPurposeVertexAnalyzer(const edm::ParameterSet &iConfig)
    : conf_(iConfig),
      ndof_(iConfig.getParameter<int>("ndof")),
      errorPrinted_(false),
      vertexInputTag_(iConfig.getParameter<edm::InputTag>("vertexLabel")),
      beamSpotInputTag_(iConfig.getParameter<edm::InputTag>("beamSpotLabel")),
      vertexToken_(consumes<reco::VertexCollection>(vertexInputTag_)),
      scoreToken_(consumes<VertexScore>(vertexInputTag_)),
      beamspotToken_(consumes<reco::BeamSpot>(beamSpotInputTag_)),
      // to be configured for each year...
      vposx_(iConfig.getParameter<double>("Xpos")),
      vposy_(iConfig.getParameter<double>("Ypos")),
      tkNoBin_(iConfig.getParameter<int>("TkSizeBin")),
      tkNoMin_(iConfig.getParameter<double>("TkSizeMin")),
      tkNoMax_(iConfig.getParameter<double>("TkSizeMax")),
      dxyBin_(iConfig.getParameter<int>("DxyBin")),
      dxyMin_(iConfig.getParameter<double>("DxyMin")),
      dxyMax_(iConfig.getParameter<double>("DxyMax")),
      dzBin_(iConfig.getParameter<int>("DzBin")),
      dzMin_(iConfig.getParameter<double>("DzMin")),
      dzMax_(iConfig.getParameter<double>("DzMax")),
      phiBin_(iConfig.getParameter<int>("PhiBin")),
      phiBin2D_(iConfig.getParameter<int>("PhiBin2D")),
      phiMin_(iConfig.getParameter<double>("PhiMin")),
      phiMax_(iConfig.getParameter<double>("PhiMax")),
      etaBin_(iConfig.getParameter<int>("EtaBin")),
      etaBin2D_(iConfig.getParameter<int>("EtaBin2D")),
      etaMin_(iConfig.getParameter<double>("EtaMin")),
      etaMax_(iConfig.getParameter<double>("EtaMax")),
      // histograms
      nbvtx(nullptr),
      bsX(nullptr),
      bsY(nullptr),
      bsZ(nullptr),
      bsSigmaZ(nullptr),
      bsDxdz(nullptr),
      bsDydz(nullptr),
      bsBeamWidthX(nullptr),
      bsBeamWidthY(nullptr),
      bsType(nullptr),
      trackpt(nullptr),
      sumpt(nullptr),
      ntracks(nullptr),
      weight(nullptr),
      chi2ndf(nullptr),
      chi2prob(nullptr),
      dxy2(nullptr),
      phi_pt1(nullptr),
      eta_pt1(nullptr),
      phi_pt10(nullptr),
      eta_pt10(nullptr) {
  usesResource(TFileService::kSharedResource);
}

//
// member functions
//

// ------------ method called for each event  ------------
void GeneralPurposeVertexAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  const auto &recVtxs = iEvent.getHandle(vertexToken_);
  const auto &scores = iEvent.getHandle(scoreToken_);
  const auto &beamSpotHandle = iEvent.getHandle(beamspotToken_);
  reco::BeamSpot beamSpot = *beamSpotHandle;

  // check for absent products and simply "return" in that case
  if (recVtxs.isValid() == false || beamSpotHandle.isValid() == false) {
    edm::LogWarning("GeneralPurposeVertexAnalyzer")
        << " Some products not available in the event: VertexCollection " << vertexInputTag_ << " " << recVtxs.isValid()
        << " BeamSpot " << beamSpotInputTag_ << " " << beamSpotHandle.isValid() << ". Skipping plots for this event";
    return;
  }

  // check upfront that refs to track are (likely) to be valid
  bool ok{true};
  for (const auto &v : *recVtxs) {
    if (v.tracksSize() > 0) {
      const auto &ref = v.trackRefAt(0);
      if (ref.isNull() || !ref.isAvailable()) {
        if (!errorPrinted_) {
          edm::LogWarning("GeneralPurposeVertexAnalyzer")
              << "Skipping vertex collection: " << vertexInputTag_
              << " since likely the track collection the vertex has refs pointing to is missing (at least the first "
                 "TrackBaseRef is null or not available)";
        } else {
          errorPrinted_ = true;
        }
        ok = false;
      }
    }
  }

  if (!ok) {
    return;
  }

  nbvtx->Fill(recVtxs->size());
  int ng = 0;
  for (auto const &vx : (*recVtxs)) {
    if (vx.isValid() && !vx.isFake() && vx.ndof() >= ndof_) {
      ++ng;
    }
  }
  nbgvtx->Fill(ng);

  if (scores.isValid() && !(*scores).empty()) {
    auto pvScore = (*scores).get(0);
    score[1]->Fill(std::sqrt(pvScore));
    for (unsigned int i = 1; i < (*scores).size(); ++i) {
      score[0]->Fill(std::sqrt((*scores).get(i)));
    }
  }

  // fill PV tracks MEs (as now, for alignment)
  if (!recVtxs->empty()) {
    vertexPlots(recVtxs->front(), beamSpot, 1);
    pvTracksPlots(recVtxs->front());

    for (reco::VertexCollection::const_iterator v = recVtxs->begin() + 1; v != recVtxs->end(); ++v) {
      vertexPlots(*v, beamSpot, 0);
    }
  }

  // Beamline plots:
  bsX->Fill(beamSpot.x0());
  bsY->Fill(beamSpot.y0());
  bsZ->Fill(beamSpot.z0());
  bsSigmaZ->Fill(beamSpot.sigmaZ());
  bsDxdz->Fill(beamSpot.dxdz());
  bsDydz->Fill(beamSpot.dydz());
  bsBeamWidthX->Fill(beamSpot.BeamWidthX() * cmToUm);
  bsBeamWidthY->Fill(beamSpot.BeamWidthY() * cmToUm);
  bsType->Fill(beamSpot.type());
}

void GeneralPurposeVertexAnalyzer::pvTracksPlots(const reco::Vertex &v) {
  if (!v.isValid())
    return;
  if (v.isFake())
    return;

  if (v.tracksSize() == 0) {
    ntracks->Fill(0);
    return;
  }

  const math::XYZPoint myVertex(v.position().x(), v.position().y(), v.position().z());

  float sumPT = 0.;
  for (const auto &t : v.tracks()) {
    const bool isHighPurity = t->quality(reco::TrackBase::highPurity);
    if (!isHighPurity) {
      continue;
    }

    const float pt = t->pt();
    trackpt->Fill(pt);

    if (pt < 1.f) {
      continue;
    }

    const float pt2 = pt * pt;
    const float eta = t->eta();
    const float phi = t->phi();
    const float w = v.trackWeight(t);
    const float chi2NDF = t->normalizedChi2();
    const float chi2Prob = TMath::Prob(t->chi2(), static_cast<int>(t->ndof()));
    const float Dxy = t->dxy(myVertex) * cmToUm;
    const float Dz = t->dz(myVertex) * cmToUm;
    const float DxyErr = t->dxyError() * cmToUm;
    const float DzErr = t->dzError() * cmToUm;

    sumPT += pt2;

    weight->Fill(w);
    chi2ndf->Fill(chi2NDF);
    chi2prob->Fill(chi2Prob);
    dxy2->Fill(Dxy);
    phi_pt1->Fill(phi);
    eta_pt1->Fill(eta);

    // dxy pT>1

    dxy_pt1.IP_->Fill(Dxy);
    dxy_pt1.IPVsPhi_->Fill(phi, Dxy);
    dxy_pt1.IPVsEta_->Fill(eta, Dxy);
    dxy_pt1.IPVsEtaVsPhi_->Fill(eta, phi, Dxy);

    dxy_pt1.IPErr_->Fill(DxyErr);
    dxy_pt1.IPPull_->Fill(Dxy / DxyErr);
    dxy_pt1.IPErrVsPhi_->Fill(phi, DxyErr);
    dxy_pt1.IPErrVsEta_->Fill(eta, DxyErr);
    dxy_pt1.IPErrVsEtaVsPhi_->Fill(eta, phi, DxyErr);

    // dz pT>1

    dz_pt1.IP_->Fill(Dz);
    dz_pt1.IPVsPhi_->Fill(phi, Dz);
    dz_pt1.IPVsEta_->Fill(eta, Dz);
    dz_pt1.IPVsEtaVsPhi_->Fill(eta, phi, Dz);

    dz_pt1.IPErr_->Fill(DzErr);
    dz_pt1.IPPull_->Fill(Dz / DzErr);
    dz_pt1.IPErrVsPhi_->Fill(phi, DzErr);
    dz_pt1.IPErrVsEta_->Fill(eta, DzErr);
    dz_pt1.IPErrVsEtaVsPhi_->Fill(eta, phi, DzErr);

    if (pt >= 10.f) {
      phi_pt10->Fill(phi);
      eta_pt10->Fill(eta);

      // dxy pT>10

      dxy_pt10.IP_->Fill(Dxy);
      dxy_pt10.IPVsPhi_->Fill(phi, Dxy);
      dxy_pt10.IPVsEta_->Fill(eta, Dxy);
      dxy_pt10.IPVsEtaVsPhi_->Fill(eta, phi, Dxy);

      dxy_pt10.IPErr_->Fill(DxyErr);
      dxy_pt10.IPPull_->Fill(Dxy / DxyErr);
      dxy_pt10.IPErrVsPhi_->Fill(phi, DxyErr);
      dxy_pt10.IPErrVsEta_->Fill(eta, DxyErr);
      dxy_pt10.IPErrVsEtaVsPhi_->Fill(eta, phi, DxyErr);

      // dz pT>10

      dz_pt10.IP_->Fill(Dz);
      dz_pt10.IPVsPhi_->Fill(phi, Dz);
      dz_pt10.IPVsEta_->Fill(eta, Dz);
      dz_pt10.IPVsEtaVsPhi_->Fill(eta, phi, Dz);

      dz_pt10.IPErr_->Fill(DzErr);
      dz_pt10.IPPull_->Fill(Dz / DzErr);
      dz_pt10.IPErrVsPhi_->Fill(phi, DzErr);
      dz_pt10.IPErrVsEta_->Fill(eta, DzErr);
      dz_pt10.IPErrVsEtaVsPhi_->Fill(eta, phi, DzErr);
    }
  }

  ntracks->Fill(static_cast<float>(v.tracks().size()));
  sumpt->Fill(sumPT);
}

void GeneralPurposeVertexAnalyzer::vertexPlots(const reco::Vertex &v, const reco::BeamSpot &beamSpot, int i) {
  if (i < 0 || i > 1)
    return;
  if (!v.isValid())
    type[i]->Fill(2.);
  else if (v.isFake())
    type[i]->Fill(1.);
  else
    type[i]->Fill(0.);

  if (v.isValid() && !v.isFake()) {
    float weight = 0;
    for (reco::Vertex::trackRef_iterator t = v.tracks_begin(); t != v.tracks_end(); t++)
      weight += v.trackWeight(*t);
    trksWeight[i]->Fill(weight);
    nbtksinvtx[i]->Fill(v.tracksSize());
    ntracksVsZ[i]->Fill(v.position().z() - beamSpot.z0(), v.tracksSize());

    vtxchi2[i]->Fill(v.chi2());
    vtxndf[i]->Fill(v.ndof());
    vtxprob[i]->Fill(ChiSquaredProbability(v.chi2(), v.ndof()));

    xrec[i]->Fill(v.position().x());
    yrec[i]->Fill(v.position().y());
    zrec[i]->Fill(v.position().z());

    float xb = beamSpot.x0() + beamSpot.dxdz() * (v.position().z() - beamSpot.z0());
    float yb = beamSpot.y0() + beamSpot.dydz() * (v.position().z() - beamSpot.z0());
    xDiff[i]->Fill((v.position().x() - xb) * cmToUm);
    yDiff[i]->Fill((v.position().y() - yb) * cmToUm);

    xerr[i]->Fill(v.xError() * cmToUm);
    yerr[i]->Fill(v.yError() * cmToUm);
    zerr[i]->Fill(v.zError() * cmToUm);
    xerrVsTrks[i]->Fill(weight, v.xError() * cmToUm);
    yerrVsTrks[i]->Fill(weight, v.yError() * cmToUm);
    zerrVsTrks[i]->Fill(weight, v.zError() * cmToUm);

    nans[i]->Fill(1., edm::isNotFinite(v.position().x()) * 1.);
    nans[i]->Fill(2., edm::isNotFinite(v.position().y()) * 1.);
    nans[i]->Fill(3., edm::isNotFinite(v.position().z()) * 1.);

    int index = 3;
    for (int k = 0; k != 3; k++) {
      for (int j = k; j != 3; j++) {
        index++;
        nans[i]->Fill(index * 1., edm::isNotFinite(v.covariance(k, j)) * 1.);
        // in addition, diagonal element must be positive
        if (j == k && v.covariance(k, j) < 0) {
          nans[i]->Fill(index * 1., 1.);
        }
      }
    }
  }
}

template <typename T, typename... Args>
T *GeneralPurposeVertexAnalyzer::book(const Args &...args) const {
  T *t = fs_->make<T>(args...);
  return t;
}

// ------------ method called once each job just before starting event loop  ------------
void GeneralPurposeVertexAnalyzer::beginJob() {
  nbvtx = book<TH1I>("vtxNbr", "Reconstructed Vertices in Event", 80, -0.5, 79.5);
  nbgvtx = book<TH1I>("goodvtxNbr", "Reconstructed Good Vertices in Event", 80, -0.5, 79.5);

  nbtksinvtx[0] = book<TH1D>("otherVtxTrksNbr", "Reconstructed Tracks in Vertex (other Vtx)", 40, -0.5, 99.5);
  ntracksVsZ[0] =
      book<TProfile>("otherVtxTrksVsZ", "Reconstructed Tracks in Vertex (other Vtx) vs Z", 80, -20., 20., 0., 100., "");
  ntracksVsZ[0]->SetXTitle("z-bs");
  ntracksVsZ[0]->SetYTitle("#tracks");

  score[0] = book<TH1D>("otherVtxScore", "sqrt(score) (other Vtx)", 100, 0., 400.);
  trksWeight[0] = book<TH1D>("otherVtxTrksWeight", "Total weight of Tracks in Vertex (other Vtx)", 40, 0, 100.);
  vtxchi2[0] = book<TH1D>("otherVtxChi2", "#chi^{2} (other Vtx)", 100, 0., 200.);
  vtxndf[0] = book<TH1D>("otherVtxNdf", "ndof (other Vtx)", 100, 0., 200.);
  vtxprob[0] = book<TH1D>("otherVtxProb", "#chi^{2} probability (other Vtx)", 100, 0., 1.);
  nans[0] = book<TH1D>("otherVtxNans", "Illegal values for x,y,z,xx,xy,xz,yy,yz,zz (other Vtx)", 9, 0.5, 9.5);

  nbtksinvtx[1] = book<TH1D>("tagVtxTrksNbr", "Reconstructed Tracks in Vertex (tagged Vtx)", 100, -0.5, 99.5);
  ntracksVsZ[1] =
      book<TProfile>("tagVtxTrksVsZ", "Reconstructed Tracks in Vertex (tagged Vtx) vs Z", 80, -20., 20., 0., 100., "");
  ntracksVsZ[1]->SetXTitle("z-bs");
  ntracksVsZ[1]->SetYTitle("#tracks");

  score[1] = book<TH1D>("tagVtxScore", "sqrt(score) (tagged Vtx)", 100, 0., 400.);
  trksWeight[1] = book<TH1D>("tagVtxTrksWeight", "Total weight of Tracks in Vertex (tagged Vtx)", 100, 0, 100.);
  vtxchi2[1] = book<TH1D>("tagVtxChi2", "#chi^{2} (tagged Vtx)", 100, 0., 200.);
  vtxndf[1] = book<TH1D>("tagVtxNdf", "ndof (tagged Vtx)", 100, 0., 200.);
  vtxprob[1] = book<TH1D>("tagVtxProb", "#chi^{2} probability (tagged Vtx)", 100, 0., 1.);
  nans[1] = book<TH1D>("tagVtxNans", "Illegal values for x,y,z,xx,xy,xz,yy,yz,zz (tagged Vtx)", 9, 0.5, 9.5);

  xrec[0] = book<TH1D>("otherPosX", "Position x Coordinate (other Vtx)", 100, vposx_ - 0.1, vposx_ + 0.1);
  yrec[0] = book<TH1D>("otherPosY", "Position y Coordinate (other Vtx)", 100, vposy_ - 0.1, vposy_ + 0.1);
  zrec[0] = book<TH1D>("otherPosZ", "Position z Coordinate (other Vtx)", 100, -20., 20.);
  xDiff[0] = book<TH1D>("otherDiffX", "X distance from BeamSpot (other Vtx)", 100, -500, 500);
  yDiff[0] = book<TH1D>("otherDiffY", "Y distance from BeamSpot (other Vtx)", 100, -500, 500);
  xerr[0] = book<TH1D>("otherErrX", "Uncertainty x Coordinate (other Vtx)", 100, 0., 100);
  yerr[0] = book<TH1D>("otherErrY", "Uncertainty y Coordinate (other Vtx)", 100, 0., 100);
  zerr[0] = book<TH1D>("otherErrZ", "Uncertainty z Coordinate (other Vtx)", 100, 0., 100);
  xerrVsTrks[0] = book<TH2D>(
      "otherErrVsWeightX", "Uncertainty x Coordinate vs. track weight (other Vtx)", 100, 0, 100., 100, 0., 100);
  yerrVsTrks[0] = book<TH2D>(
      "otherErrVsWeightY", "Uncertainty y Coordinate vs. track weight (other Vtx)", 100, 0, 100., 100, 0., 100);
  zerrVsTrks[0] = book<TH2D>(
      "otherErrVsWeightZ", "Uncertainty z Coordinate vs. track weight (other Vtx)", 100, 0, 100., 100, 0., 100);

  xrec[1] = book<TH1D>("tagPosX", "Position x Coordinate (tagged Vtx)", 100, vposx_ - 0.1, vposx_ + 0.1);
  yrec[1] = book<TH1D>("tagPosY", "Position y Coordinate (tagged Vtx)", 100, vposy_ - 0.1, vposy_ + 0.1);
  zrec[1] = book<TH1D>("tagPosZ", "Position z Coordinate (tagged Vtx)", 100, -20., 20.);
  xDiff[1] = book<TH1D>("tagDiffX", "X distance from BeamSpot (tagged Vtx)", 100, -500, 500);
  yDiff[1] = book<TH1D>("tagDiffY", "Y distance from BeamSpot (tagged Vtx)", 100, -500, 500);
  xerr[1] = book<TH1D>("tagErrX", "Uncertainty x Coordinate (tagged Vtx)", 100, 0., 100);
  yerr[1] = book<TH1D>("tagErrY", "Uncertainty y Coordinate (tagged Vtx)", 100, 0., 100);
  zerr[1] = book<TH1D>("tagErrZ", "Uncertainty z Coordinate (tagged Vtx)", 100, 0., 100);
  xerrVsTrks[1] = book<TH2D>(
      "tagErrVsWeightX", "Uncertainty x Coordinate vs. track weight (tagged Vtx)", 100, 0, 100., 100, 0., 100);
  yerrVsTrks[1] = book<TH2D>(
      "tagErrVsWeightY", "Uncertainty y Coordinate vs. track weight (tagged Vtx)", 100, 0, 100., 100, 0., 100);
  zerrVsTrks[1] = book<TH2D>(
      "tagErrVsWeightZ", "Uncertainty z Coordinate vs. track weight (tagged Vtx)", 100, 0, 100., 100, 0., 100);

  type[0] = book<TH1D>("otherType", "Vertex type (other Vtx)", 3, -0.5, 2.5);
  type[1] = book<TH1D>("tagType", "Vertex type (tagged Vtx)", 3, -0.5, 2.5);

  for (int i = 0; i < 2; ++i) {
    type[i]->GetXaxis()->SetBinLabel(1, "Valid, real");
    type[i]->GetXaxis()->SetBinLabel(2, "Valid, fake");
    type[i]->GetXaxis()->SetBinLabel(3, "Invalid");
  }

  bsX = book<TH1D>("bsX", "BeamSpot x0", 100, vposx_ - 0.1, vposx_ + 0.1);
  bsY = book<TH1D>("bsY", "BeamSpot y0", 100, vposy_ - 0.1, vposy_ + 0.1);
  bsZ = book<TH1D>("bsZ", "BeamSpot z0", 100, -2., 2.);
  bsSigmaZ = book<TH1D>("bsSigmaZ", "BeamSpot sigmaZ", 100, 0., 10.);
  bsDxdz = book<TH1D>("bsDxdz", "BeamSpot dxdz", 100, -0.0003, 0.0003);
  bsDydz = book<TH1D>("bsDydz", "BeamSpot dydz", 100, -0.0003, 0.0003);
  bsBeamWidthX = book<TH1D>("bsBeamWidthX", "BeamSpot BeamWidthX", 100, 0., 100.);
  bsBeamWidthY = book<TH1D>("bsBeamWidthY", "BeamSpot BeamWidthY", 100, 0., 100.);
  bsType = book<TH1D>("bsType", "BeamSpot type", 4, -1.5, 2.5);
  bsType->GetXaxis()->SetBinLabel(1, "Unknown");
  bsType->GetXaxis()->SetBinLabel(2, "Fake");
  bsType->GetXaxis()->SetBinLabel(3, "LHC");
  bsType->GetXaxis()->SetBinLabel(4, "Tracker");

  // repeated strings in titles
  std::string s_1 = "PV Tracks (p_{T} > 1 GeV)";
  std::string s_10 = "PV Tracks (p_{T} > 10 GeV)";

  ntracks = book<TH1D>("ntracks", fmt::sprintf("number of %s", s_1).c_str(), tkNoBin_, tkNoMin_, tkNoMax_);
  ntracks->SetXTitle(fmt::sprintf("Number of %s per Event", s_1).c_str());
  ntracks->SetYTitle("Number of Events");

  weight = book<TH1D>("weight", fmt::sprintf("weight of %s", s_1).c_str(), 100, 0., 1.);
  weight->SetXTitle(fmt::sprintf("weight of %s per Event", s_1).c_str());
  weight->SetYTitle("Number of Events");

  sumpt = book<TH1D>("sumpt", fmt::sprintf("#Sum p_{T} of %s", s_1).c_str(), 100, -0.5, 249.5);
  chi2ndf = book<TH1D>("chi2ndf", fmt::sprintf("%s #chi^{2}/ndof", s_1).c_str(), 100, 0., 20.);
  chi2prob = book<TH1D>("chi2prob", fmt::sprintf("%s #chi^{2} probability", s_1).c_str(), 100, 0., 1.);

  dxy2 = book<TH1D>("dxyzoom", fmt::sprintf("%s d_{xy} (#mum)", s_1).c_str(), dxyBin_, dxyMin_ / 5., dxyMax_ / 5.);

  trackpt = gpVertexAnalyzer::makeTH1IfLog(
      fs_, true, false, "pt_track", "PV tracks p_{T};PV tracks p_{T} [GeV];#tracks", 49, log10(1.), log10(50.));
  phi_pt1 =
      book<TH1D>("phi_pt1", fmt::sprintf("%s #phi; PV tracks #phi;#tracks", s_1).c_str(), phiBin_, phiMin_, phiMax_);
  eta_pt1 =
      book<TH1D>("eta_pt1", fmt::sprintf("%s #eta; PV tracks #eta;#tracks", s_1).c_str(), etaBin_, etaMin_, etaMax_);
  phi_pt10 =
      book<TH1D>("phi_pt10", fmt::sprintf("%s #phi; PV tracks #phi;#tracks", s_10).c_str(), phiBin_, phiMin_, phiMax_);
  eta_pt10 =
      book<TH1D>("eta_pt10", fmt::sprintf("%s #phi; PV tracks #eta;#tracks", s_10).c_str(), etaBin_, etaMin_, etaMax_);

  // initialize and book the monitors;
  dxy_pt1.varname_ = "xy";
  dxy_pt1.pTcut_ = 1.f;
  dxy_pt1.bookIPMonitor(conf_, fs_);

  dxy_pt10.varname_ = "xy";
  dxy_pt10.pTcut_ = 10.f;
  dxy_pt10.bookIPMonitor(conf_, fs_);

  dz_pt1.varname_ = "z";
  dz_pt1.pTcut_ = 1.f;
  dz_pt1.bookIPMonitor(conf_, fs_);

  dz_pt10.varname_ = "z";
  dz_pt10.pTcut_ = 10.f;
  dz_pt10.bookIPMonitor(conf_, fs_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GeneralPurposeVertexAnalyzer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("ndof", 4);
  desc.add<edm::InputTag>("vertexLabel", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("beamSpotLabel", edm::InputTag("offlineBeamSpot"));
  desc.add<double>("Xpos", 0.1);
  desc.add<double>("Ypos", 0.0);
  desc.add<int>("TkSizeBin", 100);
  desc.add<double>("TkSizeMin", 499.5);
  desc.add<double>("TkSizeMax", -0.5);
  desc.add<int>("DxyBin", 100);
  desc.add<double>("DxyMin", -2000.);
  desc.add<double>("DxyMax", 2000.);
  desc.add<int>("DzBin", 100);
  desc.add<double>("DzMin", -2000.0);
  desc.add<double>("DzMax", 2000.0);
  desc.add<int>("PhiBin", 32);
  desc.add<int>("PhiBin2D", 12);
  desc.add<double>("PhiMin", -M_PI);
  desc.add<double>("PhiMax", M_PI);
  desc.add<int>("EtaBin", 26);
  desc.add<int>("EtaBin2D", 8);
  desc.add<double>("EtaMin", -2.7);
  desc.add<double>("EtaMax", 2.7);
  desc.add<int>("PtBin", 49);
  desc.add<double>("PtMin", 1.);
  desc.add<double>("PtMax", 50.);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GeneralPurposeVertexAnalyzer);
-- dummy change --
