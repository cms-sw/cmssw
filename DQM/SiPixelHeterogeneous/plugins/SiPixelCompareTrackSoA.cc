// -*- C++ -*-
// Package:    SiPixelCompareTrackSoA
// Class:      SiPixelCompareTrackSoA
//
/**\class SiPixelCompareTrackSoA SiPixelCompareTrackSoA.cc
*/
//
// Author: Suvankar Roy Chowdhury
//
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"
// for string manipulations
#include <fmt/printf.h>

namespace {
  // same logic used for the MTV:
  // cf https://github.com/cms-sw/cmssw/blob/master/Validation/RecoTrack/src/MTVHistoProducerAlgoForTracker.cc
  typedef dqm::reco::DQMStore DQMStore;

  void setBinLog(TAxis* axis) {
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

  void setBinLogX(TH1* h) {
    TAxis* axis = h->GetXaxis();
    setBinLog(axis);
  }
  void setBinLogY(TH1* h) {
    TAxis* axis = h->GetYaxis();
    setBinLog(axis);
  }

  template <typename... Args>
  dqm::reco::MonitorElement* make2DIfLog(DQMStore::IBooker& ibook, bool logx, bool logy, Args&&... args) {
    auto h = std::make_unique<TH2I>(std::forward<Args>(args)...);
    if (logx)
      setBinLogX(h.get());
    if (logy)
      setBinLogY(h.get());
    const auto& name = h->GetName();
    return ibook.book2I(name, h.release());
  }
}  // namespace

template <typename T>
class SiPixelCompareTrackSoA : public DQMEDAnalyzer {
public:
  using PixelTrackSoA = TrackSoAHeterogeneousHost<T>;

  explicit SiPixelCompareTrackSoA(const edm::ParameterSet&);
  ~SiPixelCompareTrackSoA() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<PixelTrackSoA> tokenSoATrackCPU_;
  const edm::EDGetTokenT<PixelTrackSoA> tokenSoATrackGPU_;
  const std::string topFolderName_;
  const bool useQualityCut_;
  const pixelTrack::Quality minQuality_;
  const float dr2cut_;
  MonitorElement* hnTracks_;
  MonitorElement* hnLooseAndAboveTracks_;
  MonitorElement* hnLooseAndAboveTracks_matched_;
  MonitorElement* hnHits_;
  MonitorElement* hnHitsVsPhi_;
  MonitorElement* hnHitsVsEta_;
  MonitorElement* hnLayers_;
  MonitorElement* hnLayersVsPhi_;
  MonitorElement* hnLayersVsEta_;
  MonitorElement* hCharge_;
  MonitorElement* hchi2_;
  MonitorElement* hChi2VsPhi_;
  MonitorElement* hChi2VsEta_;
  MonitorElement* hpt_;
  MonitorElement* hptLogLog_;
  MonitorElement* heta_;
  MonitorElement* hphi_;
  MonitorElement* hz_;
  MonitorElement* htip_;
  MonitorElement* hquality_;
  //1D differences
  MonitorElement* hptdiffMatched_;
  MonitorElement* hCurvdiffMatched_;
  MonitorElement* hetadiffMatched_;
  MonitorElement* hphidiffMatched_;
  MonitorElement* hzdiffMatched_;
  MonitorElement* htipdiffMatched_;

  //for matching eff vs region: derive the ratio at harvesting
  MonitorElement* hpt_eta_tkAllCPU_;
  MonitorElement* hpt_eta_tkAllCPUMatched_;
  MonitorElement* hphi_z_tkAllCPU_;
  MonitorElement* hphi_z_tkAllCPUMatched_;
};

//
// constructors
//

template <typename T>
SiPixelCompareTrackSoA<T>::SiPixelCompareTrackSoA(const edm::ParameterSet& iConfig)
    : tokenSoATrackCPU_(consumes<PixelTrackSoA>(iConfig.getParameter<edm::InputTag>("pixelTrackSrcCPU"))),
      tokenSoATrackGPU_(consumes<PixelTrackSoA>(iConfig.getParameter<edm::InputTag>("pixelTrackSrcGPU"))),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")),
      useQualityCut_(iConfig.getParameter<bool>("useQualityCut")),
      minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
      dr2cut_(iConfig.getParameter<double>("deltaR2cut")) {}

//
// -- Analyze
//
template <typename T>
void SiPixelCompareTrackSoA<T>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using helper = TracksUtilities<T>;
  const auto& tsoaHandleCPU = iEvent.getHandle(tokenSoATrackCPU_);
  const auto& tsoaHandleGPU = iEvent.getHandle(tokenSoATrackGPU_);
  if (not tsoaHandleCPU or not tsoaHandleGPU) {
    edm::LogWarning out("SiPixelCompareTrackSoA");
    if (not tsoaHandleCPU) {
      out << "reference (cpu) tracks not found; ";
    }
    if (not tsoaHandleGPU) {
      out << "target (gpu) tracks not found; ";
    }
    out << "the comparison will not run.";
    return;
  }

  auto const& tsoaCPU = *tsoaHandleCPU;
  auto const& tsoaGPU = *tsoaHandleGPU;
  auto maxTracksCPU = tsoaCPU.view().metadata().size();  //this should be same for both?
  auto maxTracksGPU = tsoaGPU.view().metadata().size();  //this should be same for both?
  auto const* qualityCPU = tsoaCPU.view().quality();
  auto const* qualityGPU = tsoaGPU.view().quality();
  int32_t nTracksCPU = 0;
  int32_t nTracksGPU = 0;
  int32_t nLooseAndAboveTracksCPU = 0;
  int32_t nLooseAndAboveTracksCPU_matchedGPU = 0;
  int32_t nLooseAndAboveTracksGPU = 0;

  //Loop over GPU tracks and store the indices of the loose tracks. Whats happens if useQualityCut_ is false?
  std::vector<int32_t> looseTrkidxGPU;
  for (int32_t jt = 0; jt < maxTracksGPU; ++jt) {
    if (helper::nHits(tsoaGPU.view(), jt) == 0)
      break;  // this is a guard
    if (!(tsoaGPU.view()[jt].pt() > 0.))
      continue;
    nTracksGPU++;
    if (useQualityCut_ && qualityGPU[jt] < minQuality_)
      continue;
    nLooseAndAboveTracksGPU++;
    looseTrkidxGPU.emplace_back(jt);
  }

  //Now loop over CPU tracks//nested loop for loose gPU tracks
  for (int32_t it = 0; it < maxTracksCPU; ++it) {
    int nHitsCPU = helper::nHits(tsoaCPU.view(), it);

    if (nHitsCPU == 0)
      break;  // this is a guard

    float ptCPU = tsoaCPU.view()[it].pt();
    float etaCPU = tsoaCPU.view()[it].eta();
    float phiCPU = helper::phi(tsoaCPU.view(), it);
    float zipCPU = helper::zip(tsoaCPU.view(), it);
    float tipCPU = helper::tip(tsoaCPU.view(), it);

    if (!(ptCPU > 0.))
      continue;
    nTracksCPU++;
    if (useQualityCut_ && qualityCPU[it] < minQuality_)
      continue;
    nLooseAndAboveTracksCPU++;
    //Now loop over loose GPU trk and find the closest in DeltaR//do we need pt cut?
    const int32_t notFound = -1;
    int32_t closestTkidx = notFound;
    float mindr2 = dr2cut_;

    for (auto gid : looseTrkidxGPU) {
      float etaGPU = tsoaGPU.view()[gid].eta();
      float phiGPU = helper::phi(tsoaGPU.view(), gid);
      float dr2 = reco::deltaR2(etaCPU, phiCPU, etaGPU, phiGPU);
      if (dr2 > dr2cut_)
        continue;  // this is arbitrary
      if (mindr2 > dr2) {
        mindr2 = dr2;
        closestTkidx = gid;
      }
    }

    hpt_eta_tkAllCPU_->Fill(etaCPU, ptCPU);  //all CPU tk
    hphi_z_tkAllCPU_->Fill(phiCPU, zipCPU);
    if (closestTkidx == notFound)
      continue;
    nLooseAndAboveTracksCPU_matchedGPU++;

    hchi2_->Fill(tsoaCPU.view()[it].chi2(), tsoaGPU.view()[closestTkidx].chi2());
    hCharge_->Fill(helper::charge(tsoaCPU.view(), it), helper::charge(tsoaGPU.view(), closestTkidx));
    hnHits_->Fill(helper::nHits(tsoaCPU.view(), it), helper::nHits(tsoaGPU.view(), closestTkidx));
    hnLayers_->Fill(tsoaCPU.view()[it].nLayers(), tsoaGPU.view()[closestTkidx].nLayers());
    hpt_->Fill(tsoaCPU.view()[it].pt(), tsoaGPU.view()[closestTkidx].pt());
    hptLogLog_->Fill(tsoaCPU.view()[it].pt(), tsoaGPU.view()[closestTkidx].pt());
    heta_->Fill(etaCPU, tsoaGPU.view()[closestTkidx].eta());
    hphi_->Fill(etaCPU, helper::phi(tsoaGPU.view(), closestTkidx));
    hz_->Fill(zipCPU, helper::zip(tsoaGPU.view(), closestTkidx));
    htip_->Fill(tipCPU, helper::tip(tsoaGPU.view(), closestTkidx));
    hptdiffMatched_->Fill(tsoaCPU.view()[it].pt() - tsoaGPU.view()[closestTkidx].pt());
    hCurvdiffMatched_->Fill((helper::charge(tsoaCPU.view(), it) / tsoaCPU.view()[it].pt()) -
                            (helper::charge(tsoaGPU.view(), closestTkidx) / tsoaGPU.view()[closestTkidx].pt()));
    hetadiffMatched_->Fill(etaCPU - tsoaGPU.view()[closestTkidx].eta());
    hphidiffMatched_->Fill(reco::deltaPhi(etaCPU, helper::phi(tsoaGPU.view(), closestTkidx)));
    hzdiffMatched_->Fill(zipCPU - helper::zip(tsoaGPU.view(), closestTkidx));
    htipdiffMatched_->Fill(tipCPU - helper::tip(tsoaGPU.view(), closestTkidx));
    hpt_eta_tkAllCPUMatched_->Fill(etaCPU, tsoaCPU.view()[it].pt());  //matched to gpu
    hphi_z_tkAllCPUMatched_->Fill(etaCPU, zipCPU);
  }
  hnTracks_->Fill(nTracksCPU, nTracksGPU);
  hnLooseAndAboveTracks_->Fill(nLooseAndAboveTracksCPU, nLooseAndAboveTracksGPU);
  hnLooseAndAboveTracks_matched_->Fill(nLooseAndAboveTracksCPU, nLooseAndAboveTracksCPU_matchedGPU);
}

//
// -- Book Histograms
//
template <typename T>
void SiPixelCompareTrackSoA<T>::bookHistograms(DQMStore::IBooker& iBook,
                                               edm::Run const& iRun,
                                               edm::EventSetup const& iSetup) {
  iBook.cd();
  iBook.setCurrentFolder(topFolderName_);

  // clang-format off
  std::string toRep = "Number of tracks";
  // FIXME: all the 2D correlation plots are quite heavy in terms of memory consumption, so a as soon as DQM supports THnSparse
  // these should be moved to a less resource consuming format
  hnTracks_ = iBook.book2I("nTracks", fmt::sprintf("%s per event; CPU; GPU",toRep), 501, -0.5, 500.5, 501, -0.5, 500.5);
  hnLooseAndAboveTracks_ = iBook.book2I("nLooseAndAboveTracks", fmt::sprintf("%s (quality #geq loose) per event; CPU; GPU",toRep), 501, -0.5, 500.5, 501, -0.5, 500.5);
  hnLooseAndAboveTracks_matched_ = iBook.book2I("nLooseAndAboveTracks_matched", fmt::sprintf("%s (quality #geq loose) per event; CPU; GPU",toRep), 501, -0.5, 500.5, 501, -0.5, 500.5);

  toRep = "Number of all RecHits per track (quality #geq loose)";
  hnHits_ = iBook.book2I("nRecHits", fmt::sprintf("%s;CPU;GPU",toRep), 15, -0.5, 14.5, 15, -0.5, 14.5);

  toRep = "Number of all layers per track (quality #geq loose)";
  hnLayers_ = iBook.book2I("nLayers", fmt::sprintf("%s;CPU;GPU",toRep), 15, -0.5, 14.5, 15, -0.5, 14.5);

  toRep = "Track (quality #geq loose) #chi^{2}/ndof";
  hchi2_ = iBook.book2I("nChi2ndof", fmt::sprintf("%s;CPU;GPU",toRep), 40, 0., 20., 40, 0., 20.);

  toRep = "Track (quality #geq loose) charge";
  hCharge_ = iBook.book2I("charge",fmt::sprintf("%s;CPU;GPU",toRep),3, -1.5, 1.5, 3, -1.5, 1.5);

  hpt_ = iBook.book2I("pt", "Track (quality #geq loose) p_{T} [GeV];CPU;GPU", 200, 0., 200., 200, 0., 200.);
  hptLogLog_ = make2DIfLog(iBook, true, true, "ptLogLog", "Track (quality #geq loose) p_{T} [GeV];CPU;GPU", 200, log10(0.5), log10(200.), 200, log10(0.5), log10(200.));
  heta_ = iBook.book2I("eta", "Track (quality #geq loose) #eta;CPU;GPU", 30, -3., 3., 30, -3., 3.);
  hphi_ = iBook.book2I("phi", "Track (quality #geq loose) #phi;CPU;GPU", 30, -M_PI, M_PI, 30, -M_PI, M_PI);
  hz_ = iBook.book2I("z", "Track (quality #geq loose) z [cm];CPU;GPU", 30, -30., 30., 30, -30., 30.);
  htip_ = iBook.book2I("tip", "Track (quality #geq loose) TIP [cm];CPU;GPU", 100, -0.5, 0.5, 100, -0.5, 0.5);
  //1D difference plots
  hptdiffMatched_ = iBook.book1D("ptdiffmatched", " p_{T} diff [GeV] between matched tracks; #Delta p_{T} [GeV]", 60, -30., 30.);
  hCurvdiffMatched_ = iBook.book1D("curvdiffmatched", "q/p_{T} diff [GeV] between matched tracks; #Delta q/p_{T} [GeV]", 60, -30., 30.);
  hetadiffMatched_ = iBook.book1D("etadiffmatched", " #eta diff between matched tracks; #Delta #eta", 160, -0.04 ,0.04);
  hphidiffMatched_ = iBook.book1D("phidiffmatched", " #phi diff between matched tracks; #Delta #phi",  160, -0.04 ,0.04);
  hzdiffMatched_ = iBook.book1D("zdiffmatched", " z diff between matched tracks; #Delta z [cm]", 300, -1.5, 1.5);
  htipdiffMatched_ = iBook.book1D("tipdiffmatched", " TIP diff between matched tracks; #Delta TIP [cm]", 300, -1.5, 1.5);
  //2D plots for eff
  hpt_eta_tkAllCPU_ = iBook.book2I("ptetatrkAllCPU", "Track (quality #geq loose) on CPU; #eta; p_{T} [GeV];", 30, -M_PI, M_PI, 200, 0., 200.);
  hpt_eta_tkAllCPUMatched_ = iBook.book2I("ptetatrkAllCPUmatched", "Track (quality #geq loose) on CPU matched to GPU track; #eta; p_{T} [GeV];", 30, -M_PI, M_PI, 200, 0., 200.);

  hphi_z_tkAllCPU_ = iBook.book2I("phiztrkAllCPU", "Track (quality #geq loose) on CPU; #phi; z [cm];",  30, -M_PI, M_PI, 30, -30., 30.);
  hphi_z_tkAllCPUMatched_ = iBook.book2I("phiztrkAllCPUmatched", "Track (quality #geq loose) on CPU; #phi; z [cm];", 30, -M_PI, M_PI, 30, -30., 30.);

}

template<typename T>
void SiPixelCompareTrackSoA<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelTrackSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelTrackSrcCPU", edm::InputTag("pixelTracksSoA@cpu"));
  desc.add<edm::InputTag>("pixelTrackSrcGPU", edm::InputTag("pixelTracksSoA@cuda"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU");
  desc.add<bool>("useQualityCut", true);
  desc.add<std::string>("minQuality", "loose");
  desc.add<double>("deltaR2cut", 0.04);
  descriptions.addWithDefaultLabel(desc);
}

using SiPixelPhase1CompareTrackSoA = SiPixelCompareTrackSoA<pixelTopology::Phase1>;
using SiPixelPhase2CompareTrackSoA = SiPixelCompareTrackSoA<pixelTopology::Phase2>;

DEFINE_FWK_MODULE(SiPixelPhase1CompareTrackSoA);
DEFINE_FWK_MODULE(SiPixelPhase2CompareTrackSoA);
