// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      ShortenedTrackValidation
//
/*
 *\class ShortenedTrackValidation ShortenedTrackValidation.cc Alignment/OfflineValidation/plugins/ShortenedTrackValidation.cc

 Description: This module is meant to monitor the track pT resolution using the amputated tracks method, by comparing the performance using different alignments.

 Implementation: The implemenation takes advantage of the existing implementation in the DQM/TrackingMonitorSource.

*/
//
// Original Author:  Marco Musich
//         Created:  Fri, 05 Jan 2023 11:41:00 GMT
//
//

// ROOT includes files
#include "TMath.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH1I.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TLorentzVector.h"

// standard includes
#include <fmt/printf.h>

// user includes
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerMultiRecHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/transform.h"  // for edm::vector_transform

#define CREATE_HIST_1D(varname, nbins, first, last, fs) fs.make<TH1D>(#varname, #varname, nbins, first, last)

#define CREATE_HIST_2D(varname, nbins, first, last, fs) \
  fs.make<TH2D>(#varname, #varname, nbins, first, last, nbins, first, last)

const int kBPIX = PixelSubdetector::PixelBarrel;
const int kFPIX = PixelSubdetector::PixelEndcap;

class ShortenedTrackValidation : public edm::one::EDAnalyzer<edm::one::SharedResources> {
  class trackingMon {
  public:
    trackingMon() {}
    ~trackingMon() = default;

    void book(const TFileDirectory &fs) {
      h_chi2ndof = CREATE_HIST_1D(h_chi2ndof, 100, 0.0, 10.0, fs);
      h_trkQuality = CREATE_HIST_1D(h_trkQuality, 6, -1, 5, fs);
      h_trkAlgo = CREATE_HIST_1D(h_trkAlgo, reco::TrackBase::algoSize, 0.0, double(reco::TrackBase::algoSize), fs);
      h_trkOriAlgo =
          CREATE_HIST_1D(h_trkOriAlgo, reco::TrackBase::algoSize, 0.0, double(reco::TrackBase::algoSize), fs);
      h_P = CREATE_HIST_1D(h_P, 100, 0.0, 200.0, fs);
      h_Pt = CREATE_HIST_1D(h_Pt, 100, 0.0, 100.0, fs);
      h_nHit = CREATE_HIST_1D(h_nHit, 50, -0.5, 49.5, fs);
      h_nHit2D = CREATE_HIST_1D(h_nHit2D, 20, -0.5, 19.5, fs);
      h_Charge = CREATE_HIST_1D(h_Charge, 3, -1.5, 1.5, fs);
      h_QoverP = CREATE_HIST_1D(h_QoverP, 100, -1.0, 1.0, fs);
      h_QoverPZoom = CREATE_HIST_1D(h_QoverPZoom, 100, -0.1, 0.1, fs);
      h_Eta = CREATE_HIST_1D(h_Eta, 100, -3., 3., fs);
      h_Phi = CREATE_HIST_1D(h_Phi, 100, -M_PI, M_PI, fs);
      h_vx = CREATE_HIST_1D(h_vx, 100, -0.5, 0.5, fs);
      h_vy = CREATE_HIST_1D(h_vy, 100, -0.5, 0.5, fs);
      h_vz = CREATE_HIST_1D(h_vz, 100, -20.0, 20.0, fs);
      h_d0 = CREATE_HIST_1D(h_d0, 100, -0.5, 0.5, fs);
      h_dz = CREATE_HIST_1D(h_dz, 100, -20.0, 20.0, fs);
      h_dxy = CREATE_HIST_1D(h_dxy, 100, -0.5, 0.5, fs);
      h_nhpxb = CREATE_HIST_1D(h_nhpxb, 10, -0.5, 9.5, fs);
      h_nhpxe = CREATE_HIST_1D(h_nhpxe, 10, -0.5, 9.5, fs);
      h_nhTIB = CREATE_HIST_1D(h_nhTIB, 20, -0.5, 19.5, fs);
      h_nhTID = CREATE_HIST_1D(h_nhTID, 20, -0.5, 19.5, fs);
      h_nhTOB = CREATE_HIST_1D(h_nhTOB, 20, -0.5, 19.5, fs);
      h_nhTEC = CREATE_HIST_1D(h_nhTEC, 20, -0.5, 19.5, fs);
      h_dxyBS = CREATE_HIST_1D(h_dxyBS, 100, -0.05, 0.05, fs);
      h_d0BS = CREATE_HIST_1D(h_d0BS, 100, -0.05, 0.05, fs);
      h_dzBS = CREATE_HIST_1D(h_dzBS, 100, -20.0, 20., fs);
      h_dxyPV = CREATE_HIST_1D(h_dxyPV, 100, -0.05, 0.05, fs);
      h_d0PV = CREATE_HIST_1D(h_d0PV, 100, -0.05, 0.05, fs);
      h_dzPV = CREATE_HIST_1D(h_dzPV, 100, -0.05, 0.05, fs);

      edm::LogInfo("trackingMonitoring") << "done booking";
    }

    //____________________________________________________________
    int trackQual(const reco::Track &track) {
      int myquality = -99;
      if (track.quality(reco::TrackBase::undefQuality))
        myquality = -1;
      if (track.quality(reco::TrackBase::loose))
        myquality = 0;
      if (track.quality(reco::TrackBase::tight))
        myquality = 1;
      if (track.quality(reco::TrackBase::highPurity))
        myquality = 2;
      if (track.quality(reco::TrackBase::goodIterative))
        myquality = 3;

      return myquality;
    }

    //____________________________________________________________
    static bool isHit2D(const TrackingRecHit &hit) {
      if (hit.dimension() < 2) {
        return false;  // some (muon...) stuff really has RecHit1D
      } else {
        const DetId detId(hit.geographicalId());
        if (detId.det() == DetId::Tracker) {
          if (detId.subdetId() == kBPIX || detId.subdetId() == kFPIX) {
            return true;  // pixel is always 2D
          } else {        // should be SiStrip now
            if (dynamic_cast<const SiStripRecHit2D *>(&hit))
              return false;  // normal hit
            else if (dynamic_cast<const SiStripMatchedRecHit2D *>(&hit))
              return true;  // matched is 2D
            else if (dynamic_cast<const ProjectedSiStripRecHit2D *>(&hit))
              return false;  // crazy hit...
            else {
              edm::LogError("UnknownType") << "@SUB=CalibrationTrackSelector::isHit2D"
                                           << "Tracker hit not in pixel and neither SiStripRecHit2D nor "
                                           << "SiStripMatchedRecHit2D nor ProjectedSiStripRecHit2D.";
              return false;
            }
          }
        } else {  // not tracker??
          edm::LogWarning("DetectorMismatch") << "@SUB=CalibrationTrackSelector::isHit2D"
                                              << "Hit not in tracker with 'official' dimension >=2.";
          return true;  // dimension() >= 2 so accept that...
        }
      }
      // never reached...
    }

    //____________________________________________________________
    unsigned int count2DHits(const reco::Track &track) {
      unsigned int nHit2D = 0;
      for (auto iHit = track.recHitsBegin(); iHit != track.recHitsEnd(); ++iHit) {
        if (isHit2D(**iHit)) {
          ++nHit2D;
        }
      }
      return nHit2D;
    }

    //____________________________________________________________
    void fill(const reco::Track &track, const reco::BeamSpot &beamSpot, const reco::Vertex &pvtx) {
      h_chi2ndof->Fill(track.normalizedChi2());
      h_trkQuality->Fill(trackQual(track));
      h_trkAlgo->Fill(static_cast<float>(track.algo()));
      h_trkOriAlgo->Fill(static_cast<float>(track.originalAlgo()));
      h_P->Fill(track.p());
      h_Pt->Fill(track.pt());
      h_nHit->Fill(track.numberOfValidHits());
      h_nHit2D->Fill(count2DHits(track));
      h_Charge->Fill(track.charge());
      h_QoverP->Fill(track.qoverp());
      h_QoverPZoom->Fill(track.qoverp());
      h_Eta->Fill(track.eta());
      h_Phi->Fill(track.phi());
      h_vx->Fill(track.vx());
      h_vy->Fill(track.vy());
      h_vz->Fill(track.vz());
      h_d0->Fill(track.d0());
      h_dz->Fill(track.dz());
      h_dxy->Fill(track.dxy());
      h_nhpxb->Fill(track.hitPattern().numberOfValidPixelBarrelHits());
      h_nhpxe->Fill(track.hitPattern().numberOfValidPixelEndcapHits());
      h_nhTIB->Fill(track.hitPattern().numberOfValidStripTIBHits());
      h_nhTID->Fill(track.hitPattern().numberOfValidStripTIDHits());
      h_nhTOB->Fill(track.hitPattern().numberOfValidStripTOBHits());
      h_nhTEC->Fill(track.hitPattern().numberOfValidStripTECHits());

      math::XYZPoint BS(beamSpot.x0(), beamSpot.y0(), beamSpot.z0());
      h_dxyBS->Fill(track.dxy(BS));
      h_d0BS->Fill(-track.dxy(BS));
      h_dzBS->Fill(track.dz(BS));

      math::XYZPoint PV(pvtx.x(), pvtx.y(), pvtx.z());
      h_dxyPV->Fill(track.dxy(PV));
      h_d0PV->Fill(-track.dxy(PV));
      h_dzPV->Fill(track.dz(PV));
    }

  private:
    TH1D *h_chi2ndof;
    TH1D *h_trkQuality;
    TH1D *h_trkAlgo;
    TH1D *h_trkOriAlgo;
    TH1D *h_P;
    TH1D *h_Pt;
    TH1D *h_nHit;
    TH1D *h_nHit2D;
    TH1D *h_Charge;
    TH1D *h_QoverP;
    TH1D *h_QoverPZoom;
    TH1D *h_Eta;
    TH1D *h_Phi;
    TH1D *h_vx;
    TH1D *h_vy;
    TH1D *h_vz;
    TH1D *h_d0;
    TH1D *h_dz;
    TH1D *h_dxy;
    TH1D *h_nhpxb;
    TH1D *h_nhpxe;
    TH1D *h_nhTIB;
    TH1D *h_nhTID;
    TH1D *h_nhTOB;
    TH1D *h_nhTEC;
    TH1D *h_dxyBS;
    TH1D *h_d0BS;
    TH1D *h_dzBS;
    TH1D *h_dxyPV;
    TH1D *h_d0PV;
    TH1D *h_dzPV;
  };

  class trackComparator {
  public:
    trackComparator() {}
    ~trackComparator() = default;

    //__________________________________________________
    void book(const TFileDirectory &fs) {
      h2_chi2ndof = CREATE_HIST_2D(h2_chi2ndof, 100, 0.0, 10.0, fs);
      h2_trkAlgo = CREATE_HIST_2D(h2_trkAlgo, reco::TrackBase::algoSize, 0.0, double(reco::TrackBase::algoSize), fs);
      h2_trkOriAlgo =
          CREATE_HIST_2D(h2_trkOriAlgo, reco::TrackBase::algoSize, 0.0, double(reco::TrackBase::algoSize), fs);
      h2_P = CREATE_HIST_2D(h2_P, 100, 0.0, 200.0, fs);
      h2_Pt = CREATE_HIST_2D(h2_Pt, 100, 0.0, 100.0, fs);
      h2_nHit = CREATE_HIST_2D(h2_nHit, 50, -0.5, 49.5, fs);
      h2_Charge = CREATE_HIST_2D(h2_Charge, 3, -1.5, 1.5, fs);
      h2_QoverPZoom = CREATE_HIST_2D(h2_QoverPZoom, 100, -0.1, 0.1, fs);
      h2_Eta = CREATE_HIST_2D(h2_Eta, 100, -3., 3., fs);
      h2_Phi = CREATE_HIST_2D(h2_Phi, 100, -M_PI, M_PI, fs);
      h2_vx = CREATE_HIST_2D(h2_vx, 100, -0.5, 0.5, fs);
      h2_vy = CREATE_HIST_2D(h2_vy, 100, -0.5, 0.5, fs);
      h2_vz = CREATE_HIST_2D(h2_vz, 100, -20.0, 20.0, fs);
      h2_d0 = CREATE_HIST_2D(h2_d0, 100, -0.5, 0.5, fs);
      h2_dz = CREATE_HIST_2D(h2_dz, 100, -20.0, 20.0, fs);
      h2_nhpxb = CREATE_HIST_2D(h2_nhpxb, 10, -0.5, 9.5, fs);
      h2_nhpxe = CREATE_HIST_2D(h2_nhpxe, 10, -0.5, 9.5, fs);
      h2_nhTIB = CREATE_HIST_2D(h2_nhTIB, 20, -0.5, 19.5, fs);
      h2_nhTID = CREATE_HIST_2D(h2_nhTID, 20, -0.5, 19.5, fs);
      h2_nhTOB = CREATE_HIST_2D(h2_nhTOB, 20, -0.5, 19.5, fs);
      h2_nhTEC = CREATE_HIST_2D(h2_nhTEC, 20, -0.5, 19.5, fs);
    }

    //__________________________________________________
    void fill(const reco::Track &tk1, const reco::Track &tk2) {
      h2_chi2ndof->Fill(tk1.normalizedChi2(), tk2.normalizedChi2());
      h2_trkAlgo->Fill(static_cast<float>(tk1.algo()), static_cast<float>(tk2.algo()));
      h2_trkOriAlgo->Fill(static_cast<float>(tk1.originalAlgo()), static_cast<float>(tk2.originalAlgo()));
      h2_P->Fill(tk1.p(), tk2.p());
      h2_Pt->Fill(tk1.pt(), tk2.p());
      h2_nHit->Fill(tk1.numberOfValidHits(), tk2.numberOfValidHits());
      h2_Charge->Fill(tk1.charge(), tk2.charge());
      h2_QoverPZoom->Fill(tk1.qoverp(), tk2.qoverp());
      h2_Eta->Fill(tk1.eta(), tk2.eta());
      h2_Phi->Fill(tk1.phi(), tk2.phi());
      h2_vx->Fill(tk1.vx(), tk2.vx());
      h2_vy->Fill(tk1.vy(), tk2.vy());
      h2_vz->Fill(tk1.vz(), tk2.vz());
      h2_d0->Fill(tk1.d0(), tk2.d0());
      h2_dz->Fill(tk2.dz(), tk2.dz());
      h2_nhpxb->Fill(tk1.hitPattern().numberOfValidPixelBarrelHits(), tk2.hitPattern().numberOfValidPixelBarrelHits());
      h2_nhpxe->Fill(tk1.hitPattern().numberOfValidPixelEndcapHits(), tk2.hitPattern().numberOfValidPixelEndcapHits());
      h2_nhTIB->Fill(tk1.hitPattern().numberOfValidStripTIBHits(), tk2.hitPattern().numberOfValidStripTIBHits());
      h2_nhTID->Fill(tk1.hitPattern().numberOfValidStripTIDHits(), tk2.hitPattern().numberOfValidStripTIDHits());
      h2_nhTOB->Fill(tk1.hitPattern().numberOfValidStripTOBHits(), tk2.hitPattern().numberOfValidStripTOBHits());
      h2_nhTEC->Fill(tk1.hitPattern().numberOfValidStripTECHits(), tk2.hitPattern().numberOfValidStripTECHits());
    }

  private:
    TH2D *h2_chi2ndof;
    TH2D *h2_trkAlgo;
    TH2D *h2_trkOriAlgo;
    TH2D *h2_P;
    TH2D *h2_Pt;
    TH2D *h2_nHit;
    TH2D *h2_Charge;
    TH2D *h2_QoverPZoom;
    TH2D *h2_Eta;
    TH2D *h2_Phi;
    TH2D *h2_vx;
    TH2D *h2_vy;
    TH2D *h2_vz;
    TH2D *h2_d0;
    TH2D *h2_dz;
    TH2D *h2_nhpxb;
    TH2D *h2_nhpxe;
    TH2D *h2_nhTIB;
    TH2D *h2_nhTID;
    TH2D *h2_nhTOB;
    TH2D *h2_nhTEC;
  };

public:
  explicit ShortenedTrackValidation(const edm::ParameterSet &);
  ~ShortenedTrackValidation() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  template <typename T, typename... Args>
  T *book(const TFileDirectory &dir, const Args &...args) const;
  void beginJob() override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) override;

  // ----------member data ---------------------------
  edm::Service<TFileService> fs_;
  const std::string folderName_;
  const std::vector<std::string> hitsRemain_;
  const double minTracksEta_;
  const double maxTracksEta_;
  const double minTracksPt_;
  const double maxTracksPt_;

  const double maxDr_;
  const edm::InputTag tracksTag_;
  const std::vector<edm::InputTag> tracksRerecoTag_;
  const edm::InputTag BeamSpotTag_;
  const edm::InputTag VerticesTag_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracksToken_;
  const std::vector<edm::EDGetTokenT<std::vector<reco::Track>>> tracksRerecoToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  // monitoring histograms
  std::vector<TH1F *> histsPtRatioAll_;
  std::vector<TH1F *> histsPtDiffAll_;
  std::vector<TH1F *> histsEtaDiffAll_;
  std::vector<TH1F *> histsPhiDiffAll_;
  std::vector<TH2F *> histsPtRatioVsDeltaRAll_;
  std::vector<TH1F *> histsDeltaPtOverPtAll_;
  std::vector<TH1F *> histsPtAll_;
  std::vector<TH1F *> histsNhitsAll_;
  std::vector<TH1F *> histsDeltaRAll_;

  trackingMon originalTrack;
  std::vector<trackComparator *> comparators_;
  static constexpr double muMass = 0.105658;
};

// -----------------------------
// constructors and destructor
// -----------------------------
ShortenedTrackValidation::ShortenedTrackValidation(const edm::ParameterSet &ps)
    : folderName_(ps.getUntrackedParameter<std::string>("folderName", "TrackRefitting")),
      hitsRemain_(ps.getUntrackedParameter<std::vector<std::string>>("hitsRemainInput")),
      minTracksEta_(ps.getUntrackedParameter<double>("minTracksEtaInput", 0.0)),
      maxTracksEta_(ps.getUntrackedParameter<double>("maxTracksEtaInput", 2.2)),
      minTracksPt_(ps.getUntrackedParameter<double>("minTracksPtInput", 15.0)),
      maxTracksPt_(ps.getUntrackedParameter<double>("maxTracksPtInput", 99999.9)),
      maxDr_(ps.getUntrackedParameter<double>("maxDrInput", 0.01)),
      tracksTag_(ps.getUntrackedParameter<edm::InputTag>("tracksInputTag", edm::InputTag("generalTracks", "", "DQM"))),
      tracksRerecoTag_(ps.getUntrackedParameter<std::vector<edm::InputTag>>("tracksRerecoInputTag")),
      BeamSpotTag_(ps.getUntrackedParameter<edm::InputTag>("BeamSpotTag", edm::InputTag("offlineBeamSpot"))),
      VerticesTag_(ps.getUntrackedParameter<edm::InputTag>("VerticesTag", edm::InputTag("offlinePrimaryVertices"))),
      tracksToken_(consumes<std::vector<reco::Track>>(tracksTag_)),
      tracksRerecoToken_(edm::vector_transform(
          tracksRerecoTag_, [this](edm::InputTag const &tag) { return consumes<std::vector<reco::Track>>(tag); })),
      beamspotToken_(consumes<reco::BeamSpot>(BeamSpotTag_)),
      vertexToken_(consumes<reco::VertexCollection>(VerticesTag_)) {
  usesResource(TFileService::kSharedResource);
  histsPtRatioAll_.clear();
  histsPtDiffAll_.clear();
  histsEtaDiffAll_.clear();
  histsPhiDiffAll_.clear();
  histsPtRatioVsDeltaRAll_.clear();
  histsDeltaPtOverPtAll_.clear();
  histsPtAll_.clear();
  histsNhitsAll_.clear();
  histsDeltaRAll_.clear();
  comparators_.clear();

  comparators_.reserve(hitsRemain_.size());
  for (unsigned int i = 0; i < hitsRemain_.size(); ++i) {
    comparators_.push_back(new trackComparator());
  }
}

//__________________________________________________________________________________
template <typename T, typename... Args>
T *ShortenedTrackValidation::book(const TFileDirectory &dir, const Args &...args) const {
  T *t = dir.make<T>(args...);
  return t;
}

//__________________________________________________________________________________
void ShortenedTrackValidation::beginJob() {
  std::string currentFolder = folderName_ + "/Resolutions";
  TFileDirectory ShortTrackResolution = fs_->mkdir(currentFolder);
  currentFolder = folderName_ + "/Tracks";
  TFileDirectory TrackQuals = fs_->mkdir(currentFolder);

  for (unsigned int i = 0; i < hitsRemain_.size(); ++i) {
    histsPtRatioAll_.push_back(
        book<TH1F>(ShortTrackResolution,
                   fmt::sprintf("trackPtRatio_%s", hitsRemain_[i]).c_str(),
                   fmt::sprintf("Short Track p_{T} / Full Track p_{T} - %s layers;p_{T}^{short}/p_{T}^{full};n. tracks",
                                hitsRemain_[i])
                       .c_str(),
                   100,
                   0.5,
                   1.5));

    histsPtDiffAll_.push_back(book<TH1F>(
        ShortTrackResolution,
        fmt::sprintf("trackPtDiff_%s", hitsRemain_[i]).c_str(),
        fmt::sprintf("Short Track p_{T} - Full Track p_{T} - %s layers;p_{T}^{short} - p_{T}^{full} [GeV];n. tracks",
                     hitsRemain_[i])
            .c_str(),
        100,
        -10.,
        10.));

    histsEtaDiffAll_.push_back(
        book<TH1F>(ShortTrackResolution,
                   fmt::sprintf("trackEtaDiff_%s", hitsRemain_[i]).c_str(),
                   fmt::sprintf("Short Track #eta - Full Track #eta - %s layers;#eta^{short} - #eta^{full};n. tracks",
                                hitsRemain_[i])
                       .c_str(),
                   100,
                   -0.001,
                   0.001));

    histsPhiDiffAll_.push_back(
        book<TH1F>(ShortTrackResolution,
                   fmt::sprintf("trackPhiDiff_%s", hitsRemain_[i]).c_str(),
                   fmt::sprintf("Short Track #phi - Full Track #phi - %s layers;#phi^{short} - #phi^{full};n. tracks",
                                hitsRemain_[i])
                       .c_str(),
                   100,
                   -0.001,
                   0.001));

    histsPtRatioVsDeltaRAll_.push_back(
        book<TH2F>(ShortTrackResolution,
                   fmt::sprintf("trackPtRatioVsDeltaR_%s", hitsRemain_[i]).c_str(),
                   fmt::sprintf("Short Track p_{T} / Full Track p_{T} - %s layers vs "
                                "#DeltaR;#DeltaR(short,full);p_{T}^{short}/p_{T}^{full} [GeV];n. tracks",
                                hitsRemain_[i])
                       .c_str(),
                   100,
                   0.,
                   0.01,
                   101,
                   -0.05,
                   2.05));

    histsDeltaPtOverPtAll_.push_back(
        book<TH1F>(ShortTrackResolution,
                   fmt::sprintf("trackDeltaPtOverPt_%s", hitsRemain_[i]).c_str(),
                   fmt::sprintf("Short Track p_{T} - Full Track p_{T} / Full Track p_{T} - %s layers;p_{T}^{short} - "
                                "p_{T}^{full} / p^{full}_{T};n. tracks",
                                hitsRemain_[i])
                       .c_str(),
                   101,
                   -5.,
                   5.));

    histsPtAll_.push_back(
        book<TH1F>(TrackQuals,
                   fmt::sprintf("trackPt_%s", hitsRemain_[i]).c_str(),
                   fmt::sprintf("Short Track p_{T} - %s layers;p_{T}^{short} [GeV];n. tracks", hitsRemain_[i]).c_str(),
                   100,
                   0.,
                   100.));

    histsNhitsAll_.push_back(
        book<TH1F>(TrackQuals,
                   fmt::sprintf("trackNhits_%s", hitsRemain_[i]).c_str(),
                   fmt::sprintf("Short Track n. hits - %s layers; n. hits per track;n. tracks", hitsRemain_[i]).c_str(),
                   20,
                   -0.5,
                   19.5));

    histsDeltaRAll_.push_back(book<TH1F>(
        TrackQuals,
        fmt::sprintf("trackDeltaR_%s", hitsRemain_[i]).c_str(),
        fmt::sprintf("Short Track / Long Track #DeltaR %s layers;#DeltaR(short,long);n. tracks", hitsRemain_[i]).c_str(),
        100,
        0.,
        0.005));

    currentFolder = fmt::sprintf("%s/Compare_%sHit", folderName_, hitsRemain_[i]);
    comparators_[i]->book(fs_->mkdir(currentFolder));
  }

  currentFolder = folderName_ + "/OriginalTrack";
  TFileDirectory original = fs_->mkdir(currentFolder);
  originalTrack.book(original);
}

//__________________________________________________________________________________
void ShortenedTrackValidation::analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
  const auto &tracks = iEvent.getHandle(tracksToken_);

  if (!tracks.isValid()) {
    edm::LogError("ShortenedTrackValidation") << "Missing input track collection " << tracksTag_.encode() << std::endl;
    return;
  }

  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> beamSpotHandle = iEvent.getHandle(beamspotToken_);
  if (beamSpotHandle.isValid()) {
    beamSpot = *beamSpotHandle;
  } else {
    beamSpot = reco::BeamSpot();
  }

  reco::Vertex pvtx;
  edm::Handle<reco::VertexCollection> vertexHandle = iEvent.getHandle(vertexToken_);
  if (vertexHandle.isValid()) {
    pvtx = (*vertexHandle).at(0);
  } else {
    pvtx = reco::Vertex();
  }

  // the original long track
  for (const auto &track : *tracks) {
    const reco::HitPattern &hp = track.hitPattern();
    if (int(int(hp.numberOfValidHits()) - int(hp.numberOfAllHits(reco::HitPattern::TRACK_HITS))) != 0) {
      break;
    }

    // fill the original track properties monitoring
    originalTrack.fill(track, beamSpot, pvtx);

    TLorentzVector tvec;
    tvec.SetPtEtaPhiM(track.pt(), track.eta(), track.phi(), muMass);

    int i = 0;  // token index
    // loop on the re-recoed shortened track collections
    for (const auto &token : tracksRerecoToken_) {
      const auto &tracks_rereco = iEvent.getHandle(token);

      for (const auto &track_rereco : *tracks_rereco) {
        TLorentzVector trerecovec;
        trerecovec.SetPtEtaPhiM(track_rereco.pt(), track_rereco.eta(), track_rereco.phi(), 0.0);
        double deltaR = tvec.DeltaR(trerecovec);

        if (deltaR < maxDr_) {
          if (track_rereco.pt() >= minTracksPt_ && track_rereco.pt() <= maxTracksPt_ &&
              std::abs(track_rereco.eta()) >= minTracksEta_ && std::abs(track_rereco.eta()) <= maxTracksEta_) {
            // fill the 2D comparisons per track
            comparators_[i]->fill(track, track_rereco);

            histsPtRatioAll_[i]->Fill(1.0 * track_rereco.pt() / track.pt());
            histsPtDiffAll_[i]->Fill(track_rereco.pt() - track.pt());
            histsDeltaPtOverPtAll_[i]->Fill((track_rereco.pt() - track.pt()) / track.pt());
            histsEtaDiffAll_[i]->Fill(track_rereco.eta() - track.eta());
            histsPhiDiffAll_[i]->Fill(track_rereco.phi() - track.phi());
            histsPtRatioVsDeltaRAll_[i]->Fill(deltaR, track_rereco.pt() / track.pt());
            histsPtAll_[i]->Fill(track_rereco.pt());
            histsNhitsAll_[i]->Fill(track_rereco.numberOfValidHits());
            histsDeltaRAll_[i]->Fill(deltaR);
          }
        }
      }
      ++i;
    }
  }
}

//__________________________________________________________________________________
void ShortenedTrackValidation::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("folderName", "TrackRefitting");
  desc.addUntracked<std::vector<std::string>>("hitsRemainInput", {});
  desc.addUntracked<double>("minTracksEtaInput", 0.0);
  desc.addUntracked<double>("maxTracksEtaInput", 2.2);
  desc.addUntracked<double>("minTracksPtInput", 15.0);
  desc.addUntracked<double>("maxTracksPtInput", 99999.9);
  desc.addUntracked<double>("maxDrInput", 0.01);
  desc.addUntracked<edm::InputTag>("tracksInputTag", edm::InputTag("generalTracks", "", "DQM"));
  desc.addUntracked<std::vector<edm::InputTag>>("tracksRerecoInputTag", {});
  desc.addUntracked<edm::InputTag>("BeamSpotTag", edm::InputTag("offlineBeamSpot"));
  desc.addUntracked<edm::InputTag>("VerticesTag", edm::InputTag("offlinePrimaryVertices"));
  descriptions.addWithDefaultLabel(desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ShortenedTrackValidation);
-- dummy change --
