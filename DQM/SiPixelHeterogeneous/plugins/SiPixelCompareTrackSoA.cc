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
#include "Geometry/CommonTopologies/interface/SimpleSeedingLayersTopology.h"
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
  MonitorElement* hDeltaNTracks_;
  MonitorElement* hDeltaNLooseAndAboveTracks_;
  MonitorElement* hDeltaNLooseAndAboveTracks_matched_;
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
  MonitorElement* hCurvature_;
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
  MonitorElement* hpt_eta_tkAllRef_;
  MonitorElement* hpt_eta_tkAllRefMatched_;
  MonitorElement* hphi_z_tkAllRef_;
  MonitorElement* hphi_z_tkAllRefMatched_;
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
    auto qCPU = helper::charge(tsoaCPU.view(), it);

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

    hpt_eta_tkAllRef_->Fill(etaCPU, ptCPU);  //all CPU tk
    hphi_z_tkAllRef_->Fill(phiCPU, zipCPU);
    if (closestTkidx == notFound)
      continue;
    nLooseAndAboveTracksCPU_matchedGPU++;

    hchi2_->Fill(tsoaCPU.view()[it].chi2(), tsoaGPU.view()[closestTkidx].chi2());
    hCharge_->Fill(qCPU, helper::charge(tsoaGPU.view(), closestTkidx));
    hnHits_->Fill(helper::nHits(tsoaCPU.view(), it), helper::nHits(tsoaGPU.view(), closestTkidx));
    hnLayers_->Fill(tsoaCPU.view()[it].nLayers(), tsoaGPU.view()[closestTkidx].nLayers());
    hpt_->Fill(ptCPU, tsoaGPU.view()[closestTkidx].pt());
    hCurvature_->Fill(qCPU / ptCPU, helper::charge(tsoaGPU.view(), closestTkidx) / tsoaGPU.view()[closestTkidx].pt());
    hptLogLog_->Fill(ptCPU, tsoaGPU.view()[closestTkidx].pt());
    heta_->Fill(etaCPU, tsoaGPU.view()[closestTkidx].eta());
    hphi_->Fill(phiCPU, helper::phi(tsoaGPU.view(), closestTkidx));
    hz_->Fill(zipCPU, helper::zip(tsoaGPU.view(), closestTkidx));
    htip_->Fill(tipCPU, helper::tip(tsoaGPU.view(), closestTkidx));
    hptdiffMatched_->Fill(ptCPU - tsoaGPU.view()[closestTkidx].pt());
    hCurvdiffMatched_->Fill((helper::charge(tsoaCPU.view(), it) / tsoaCPU.view()[it].pt()) -
                            (helper::charge(tsoaGPU.view(), closestTkidx) / tsoaGPU.view()[closestTkidx].pt()));
    hetadiffMatched_->Fill(etaCPU - tsoaGPU.view()[closestTkidx].eta());
    hphidiffMatched_->Fill(reco::deltaPhi(phiCPU, helper::phi(tsoaGPU.view(), closestTkidx)));
    hzdiffMatched_->Fill(zipCPU - helper::zip(tsoaGPU.view(), closestTkidx));
    htipdiffMatched_->Fill(tipCPU - helper::tip(tsoaGPU.view(), closestTkidx));
    hpt_eta_tkAllRefMatched_->Fill(etaCPU, tsoaCPU.view()[it].pt());  //matched to gpu
    hphi_z_tkAllRefMatched_->Fill(etaCPU, zipCPU);
  }

  // Define a lambda function for filling the histograms
  auto fillHistogram = [](auto& histogram, auto xValue, auto yValue) { histogram->Fill(xValue, yValue); };

  // Define a lambda for filling delta histograms
  auto fillDeltaHistogram = [](auto& histogram, int cpuValue, int gpuValue) {
    histogram->Fill(std::min(cpuValue, 1000), std::clamp(gpuValue - cpuValue, -100, 100));
  };

  // Fill the histograms
  fillHistogram(hnTracks_, nTracksCPU, nTracksGPU);
  fillHistogram(hnLooseAndAboveTracks_, nLooseAndAboveTracksCPU, nLooseAndAboveTracksGPU);
  fillHistogram(hnLooseAndAboveTracks_matched_, nLooseAndAboveTracksCPU, nLooseAndAboveTracksCPU_matchedGPU);

  fillDeltaHistogram(hDeltaNTracks_, nTracksCPU, nTracksGPU);
  fillDeltaHistogram(hDeltaNLooseAndAboveTracks_, nLooseAndAboveTracksCPU, nLooseAndAboveTracksGPU);
  fillDeltaHistogram(hDeltaNLooseAndAboveTracks_matched_, nLooseAndAboveTracksCPU, nLooseAndAboveTracksCPU_matchedGPU);
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

  // Define a helper function for booking histograms
  std::string toRep = "Number of tracks";
  auto bookTracksTH2I = [&](const std::string& name,
                            const std::string& title,
                            int xBins,
                            double xMin,
                            double xMax,
                            int yBins,
                            double yMin,
                            double yMax) {
    return iBook.book2I(name, fmt::sprintf(title, toRep), xBins, xMin, xMax, yBins, yMin, yMax);
  };

  // Define common parameters for different histogram types
  constexpr int xBins = 501;
  constexpr double xMin = -0.5;
  constexpr double xMax = 1001.5;

  constexpr int dXBins = 1001;
  constexpr double dXMin = -0.5;
  constexpr double dXMax = 1000.5;

  constexpr int dYBins = 201;
  constexpr double dYMin = -100.5;
  constexpr double dYMax = 100.5;

  // FIXME: all the 2D correlation plots are quite heavy in terms of memory consumption, so a as soon as DQM supports THnSparse
  // these should be moved to a less resource consuming format

  // Book histograms using the helper function
  // clang-format off
  hnTracks_ = bookTracksTH2I("nTracks", "%s per event; Reference; Target", xBins, xMin, xMax, xBins, xMin, xMax);
  hnLooseAndAboveTracks_ = bookTracksTH2I("nLooseAndAboveTracks", "%s (quality #geq loose) per event; Reference; Target", xBins, xMin, xMax, xBins, xMin, xMax);
  hnLooseAndAboveTracks_matched_ = bookTracksTH2I("nLooseAndAboveTracks_matched", "%s (quality #geq loose) per event; Reference; Target", xBins, xMin, xMax, xBins, xMin, xMax);

  hDeltaNTracks_ = bookTracksTH2I("deltaNTracks", "%s per event; Reference; Target - Reference", dXBins, dXMin, dXMax, dYBins, dYMin, dYMax);
  hDeltaNLooseAndAboveTracks_ = bookTracksTH2I("deltaNLooseAndAboveTracks", "%s (quality #geq loose) per event; Reference; Target - Reference", dXBins, dXMin, dXMax, dYBins, dYMin, dYMax);
  hDeltaNLooseAndAboveTracks_matched_ = bookTracksTH2I("deltaNLooseAndAboveTracks_matched", "%s (quality #geq loose) per event; Reference; Target - Reference", dXBins, dXMin, dXMax, dYBins, dYMin, dYMax);

  toRep = "Number of all RecHits per track (quality #geq loose)";
  hnHits_ = iBook.book2I("nRecHits", fmt::sprintf("%s;CPU;GPU",toRep), 15, -0.5, 14.5, 15, -0.5, 14.5);

  toRep = "Number of all layers per track (quality #geq loose)";
  hnLayers_ = iBook.book2I("nLayers", fmt::sprintf("%s;CPU;GPU",toRep), 15, -0.5, 14.5, 15, -0.5, 14.5);

  toRep = "Track (quality #geq loose) #chi^{2}/ndof";
  hchi2_ = iBook.book2I("nChi2ndof", fmt::sprintf("%s;CPU;GPU",toRep), 40, 0., 20., 40, 0., 20.);

  toRep = "Track (quality #geq loose) charge";
  hCharge_ = iBook.book2I("charge",fmt::sprintf("%s;CPU;GPU",toRep),3, -1.5, 1.5, 3, -1.5, 1.5);

  hpt_ = iBook.book2I("pt", "Track (quality #geq loose) p_{T} [GeV];CPU;GPU", 200, 0., 200., 200, 0., 200.);
  hCurvature_ = iBook.book2I("curvature", "Track (quality #geq loose) q/p_{T} [GeV^{-1}];CPU;GPU",  60,- 3., 3., 60, -3., 3. );
  hptLogLog_ = make2DIfLog(iBook, true, true, "ptLogLog", "Track (quality #geq loose) p_{T} [GeV];CPU;GPU", 200, log10(0.5), log10(200.), 200, log10(0.5), log10(200.));
  heta_ = iBook.book2I("eta", "Track (quality #geq loose) #eta;CPU;GPU", 30, -3., 3., 30, -3., 3.);
  hphi_ = iBook.book2I("phi", "Track (quality #geq loose) #phi;CPU;GPU", 30, -M_PI, M_PI, 30, -M_PI, M_PI);
  hz_ = iBook.book2I("z", "Track (quality #geq loose) z [cm];CPU;GPU", 30, -30., 30., 30, -30., 30.);
  htip_ = iBook.book2I("tip", "Track (quality #geq loose) TIP [cm];CPU;GPU", 100, -0.5, 0.5, 100, -0.5, 0.5);
  //1D difference plots
  hptdiffMatched_ = iBook.book1D("ptdiffmatched", " p_{T} diff [GeV] between matched tracks; #Delta p_{T} [GeV]", 61, -30.5, 30.5);
  hCurvdiffMatched_ = iBook.book1D("curvdiffmatched", "q/p_{T} diff [GeV^{-1}] between matched tracks; #Delta q/p_{T} [GeV^{-1}]", 61, -3.05, 3.05);
  hetadiffMatched_ = iBook.book1D("etadiffmatched", " #eta diff between matched tracks; #Delta #eta", 161, -0.045 ,0.045);
  hphidiffMatched_ = iBook.book1D("phidiffmatched", " #phi diff between matched tracks; #Delta #phi",  161, -0.045 ,0.045);
  hzdiffMatched_ = iBook.book1D("zdiffmatched", " z diff between matched tracks; #Delta z [cm]", 301, -1.55, 1.55);
  htipdiffMatched_ = iBook.book1D("tipdiffmatched", " TIP diff between matched tracks; #Delta TIP [cm]", 301, -1.55, 1.55);
  //2D plots for eff
  hpt_eta_tkAllRef_ = iBook.book2I("ptetatrkAllReference", "Track (quality #geq loose) on CPU; #eta; p_{T} [GeV];", 30, -M_PI, M_PI, 200, 0., 200.);
  hpt_eta_tkAllRefMatched_ = iBook.book2I("ptetatrkAllReferencematched", "Track (quality #geq loose) on CPU matched to GPU track; #eta; p_{T} [GeV];", 30, -M_PI, M_PI, 200, 0., 200.);

  hphi_z_tkAllRef_ = iBook.book2I("phiztrkAllReference", "Track (quality #geq loose) on CPU; #phi; z [cm];",  30, -M_PI, M_PI, 30, -30., 30.);
  hphi_z_tkAllRefMatched_ = iBook.book2I("phiztrkAllReferencematched", "Track (quality #geq loose) on CPU; #phi; z [cm];", 30, -M_PI, M_PI, 30, -30., 30.);

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
  desc.add<double>("deltaR2cut", 0.02 * 0.02)->setComment("deltaR2 cut between track on CPU and GPU");
  descriptions.addWithDefaultLabel(desc);
}

using SiPixelPhase1CompareTrackSoA = SiPixelCompareTrackSoA<pixelTopology::Phase1>;
using SiPixelPhase2CompareTrackSoA = SiPixelCompareTrackSoA<pixelTopology::Phase2>;
using SiPixelHIonPhase1CompareTrackSoA = SiPixelCompareTrackSoA<pixelTopology::HIonPhase1>;
using SiPixelPhase1StripCompareTrackSoA = SiPixelCompareTrackSoA<pixelTopology::Phase1Strip>;

DEFINE_FWK_MODULE(SiPixelPhase1CompareTrackSoA);
DEFINE_FWK_MODULE(SiPixelPhase1StripCompareTrackSoA);
DEFINE_FWK_MODULE(SiPixelPhase2CompareTrackSoA);
DEFINE_FWK_MODULE(SiPixelHIonPhase1CompareTrackSoA);
