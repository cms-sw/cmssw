// TODO: change file name to SiPixelCompareTracksSoA.cc when CUDA code is removed

// -*- C++ -*-
// Package:    SiPixelCompareTracks
// Class:      SiPixelCompareTracks
//
/**\class SiPixelCompareTracks SiPixelCompareTracks.cc
*/
//
// Author: Suvankar Roy Chowdhury
//

// for string manipulations
#include <fmt/printf.h>
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
// DataFormats
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"

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

// TODO: change class name to SiPixelCompareTracksSoA when CUDA code is removed
template <typename T>
class SiPixelCompareTracks : public DQMEDAnalyzer {
public:
  using PixelTrackSoA = TracksHost<T>;

  explicit SiPixelCompareTracks(const edm::ParameterSet&);
  ~SiPixelCompareTracks() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  // analyzeSeparate is templated to accept distinct types of SoAs
  // The default use case is to use tracks from Alpaka reconstructed on CPU and GPU;
  template <typename U, typename V>
  void analyzeSeparate(U tokenRef, V tokenTar, const edm::Event& iEvent);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // these two are both on Host but originally they have been produced on Host or on Device
  const edm::EDGetTokenT<PixelTrackSoA> tokenSoATrackReference_;
  const edm::EDGetTokenT<PixelTrackSoA> tokenSoATrackTarget_;
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
  MonitorElement* hpt_eta_tkAllRef_;
  MonitorElement* hpt_eta_tkAllRefMatched_;
  MonitorElement* hphi_z_tkAllRef_;
  MonitorElement* hphi_z_tkAllRefMatched_;
};

//
// constructors
//

template <typename T>
SiPixelCompareTracks<T>::SiPixelCompareTracks(const edm::ParameterSet& iConfig)
    : tokenSoATrackReference_(consumes<PixelTrackSoA>(iConfig.getParameter<edm::InputTag>("pixelTrackReferenceSoA"))),
      tokenSoATrackTarget_(consumes<PixelTrackSoA>(iConfig.getParameter<edm::InputTag>("pixelTrackTargetSoA"))),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")),
      useQualityCut_(iConfig.getParameter<bool>("useQualityCut")),
      minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
      dr2cut_(iConfig.getParameter<double>("deltaR2cut")) {}

template <typename T>
template <typename U, typename V>
void SiPixelCompareTracks<T>::analyzeSeparate(U tokenRef, V tokenTar, const edm::Event& iEvent) {
  using helper = TracksUtilities<T>;

  const auto& tsoaHandleRef = iEvent.getHandle(tokenRef);
  const auto& tsoaHandleTar = iEvent.getHandle(tokenTar);

  if (not tsoaHandleRef or not tsoaHandleTar) {
    edm::LogWarning out("SiPixelCompareTracks");
    if (not tsoaHandleRef) {
      out << "reference tracks not found; ";
    }
    if (not tsoaHandleTar) {
      out << "target tracks not found; ";
    }
    out << "the comparison will not run.";
    return;
  }

  auto const& tsoaRef = *tsoaHandleRef;
  auto const& tsoaTar = *tsoaHandleTar;

  auto maxTracksRef = tsoaRef.view().metadata().size();  //this should be same for both?
  auto maxTracksTar = tsoaTar.view().metadata().size();  //this should be same for both?

  auto const* qualityRef = tsoaRef.view().quality();
  auto const* qualityTar = tsoaTar.view().quality();

  int32_t nTracksRef = 0;
  int32_t nTracksTar = 0;
  int32_t nLooseAndAboveTracksRef = 0;
  int32_t nLooseAndAboveTracksRef_matchedTar = 0;
  int32_t nLooseAndAboveTracksTar = 0;

  //Loop over Tar tracks and store the indices of the loose tracks. Whats happens if useQualityCut_ is false?
  std::vector<int32_t> looseTrkidxTar;
  for (int32_t jt = 0; jt < maxTracksTar; ++jt) {
    if (helper::nHits(tsoaTar.view(), jt) == 0)
      break;  // this is a guard
    if (!(tsoaTar.view()[jt].pt() > 0.))
      continue;
    nTracksTar++;
    if (useQualityCut_ && qualityTar[jt] < minQuality_)
      continue;
    nLooseAndAboveTracksTar++;
    looseTrkidxTar.emplace_back(jt);
  }

  //Now loop over Ref tracks//nested loop for loose gPU tracks
  for (int32_t it = 0; it < maxTracksRef; ++it) {
    int nHitsRef = helper::nHits(tsoaRef.view(), it);

    if (nHitsRef == 0)
      break;  // this is a guard

    float ptRef = tsoaRef.view()[it].pt();
    float etaRef = tsoaRef.view()[it].eta();
    float phiRef = reco::phi(tsoaRef.view(), it);
    float zipRef = reco::zip(tsoaRef.view(), it);
    float tipRef = reco::tip(tsoaRef.view(), it);

    if (!(ptRef > 0.))
      continue;
    nTracksRef++;
    if (useQualityCut_ && qualityRef[it] < minQuality_)
      continue;
    nLooseAndAboveTracksRef++;
    //Now loop over loose Tar trk and find the closest in DeltaR//do we need pt cut?
    const int32_t notFound = -1;
    int32_t closestTkidx = notFound;
    float mindr2 = dr2cut_;

    for (auto gid : looseTrkidxTar) {
      float etaTar = tsoaTar.view()[gid].eta();
      float phiTar = reco::phi(tsoaTar.view(), gid);
      float dr2 = reco::deltaR2(etaRef, phiRef, etaTar, phiTar);
      if (dr2 > dr2cut_)
        continue;  // this is arbitrary
      if (mindr2 > dr2) {
        mindr2 = dr2;
        closestTkidx = gid;
      }
    }

    hpt_eta_tkAllRef_->Fill(etaRef, ptRef);  //all Ref tk
    hphi_z_tkAllRef_->Fill(phiRef, zipRef);
    if (closestTkidx == notFound)
      continue;
    nLooseAndAboveTracksRef_matchedTar++;

    hchi2_->Fill(tsoaRef.view()[it].chi2(), tsoaTar.view()[closestTkidx].chi2());
    hCharge_->Fill(reco::charge(tsoaRef.view(), it), reco::charge(tsoaTar.view(), closestTkidx));
    hnHits_->Fill(helper::nHits(tsoaRef.view(), it), helper::nHits(tsoaTar.view(), closestTkidx));
    hnLayers_->Fill(tsoaRef.view()[it].nLayers(), tsoaTar.view()[closestTkidx].nLayers());
    hpt_->Fill(ptRef, tsoaTar.view()[closestTkidx].pt());
    hptLogLog_->Fill(ptRef, tsoaTar.view()[closestTkidx].pt());
    heta_->Fill(etaRef, tsoaTar.view()[closestTkidx].eta());
    hphi_->Fill(phiRef, reco::phi(tsoaTar.view(), closestTkidx));
    hz_->Fill(zipRef, reco::zip(tsoaTar.view(), closestTkidx));
    htip_->Fill(tipRef, reco::tip(tsoaTar.view(), closestTkidx));
    hptdiffMatched_->Fill(ptRef - tsoaTar.view()[closestTkidx].pt());
    hCurvdiffMatched_->Fill((reco::charge(tsoaRef.view(), it) / tsoaRef.view()[it].pt()) -
                            (reco::charge(tsoaTar.view(), closestTkidx) / tsoaTar.view()[closestTkidx].pt()));
    hetadiffMatched_->Fill(etaRef - tsoaTar.view()[closestTkidx].eta());
    hphidiffMatched_->Fill(reco::deltaPhi(phiRef, reco::phi(tsoaTar.view(), closestTkidx)));
    hzdiffMatched_->Fill(zipRef - reco::zip(tsoaTar.view(), closestTkidx));
    htipdiffMatched_->Fill(tipRef - reco::tip(tsoaTar.view(), closestTkidx));
    hpt_eta_tkAllRefMatched_->Fill(etaRef, tsoaRef.view()[it].pt());  //matched to gpu
    hphi_z_tkAllRefMatched_->Fill(etaRef, zipRef);
  }
  hnTracks_->Fill(nTracksRef, nTracksTar);
  hnLooseAndAboveTracks_->Fill(nLooseAndAboveTracksRef, nLooseAndAboveTracksTar);
  hnLooseAndAboveTracks_matched_->Fill(nLooseAndAboveTracksRef, nLooseAndAboveTracksRef_matchedTar);
}

//
// -- Analyze
//
template <typename T>
void SiPixelCompareTracks<T>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // The default use case is to use vertices from Alpaka reconstructed on CPU and GPU;
  // The function is left templated if any other cases need to be added
  analyzeSeparate(tokenSoATrackReference_, tokenSoATrackTarget_, iEvent);
}

//
// -- Book Histograms
//
template <typename T>
void SiPixelCompareTracks<T>::bookHistograms(DQMStore::IBooker& iBook,
                                             edm::Run const& iRun,
                                             edm::EventSetup const& iSetup) {
  iBook.cd();
  iBook.setCurrentFolder(topFolderName_);

  // clang-format off
  std::string toRep = "Number of tracks";
  // FIXME: all the 2D correlation plots are quite heavy in terms of memory consumption, so a as soon as DQM supports THnSparse
  // these should be moved to a less resource consuming format
  hnTracks_ = iBook.book2I("nTracks", fmt::sprintf("%s per event; Reference; Target",toRep), 501, -0.5, 500.5, 501, -0.5, 500.5);
  hnLooseAndAboveTracks_ = iBook.book2I("nLooseAndAboveTracks", fmt::sprintf("%s (quality #geq loose) per event; Reference; Target",toRep), 501, -0.5, 500.5, 501, -0.5, 500.5);
  hnLooseAndAboveTracks_matched_ = iBook.book2I("nLooseAndAboveTracks_matched", fmt::sprintf("%s (quality #geq loose) per event; Reference; Target",toRep), 501, -0.5, 500.5, 501, -0.5, 500.5);

  toRep = "Number of all RecHits per track (quality #geq loose)";
  hnHits_ = iBook.book2I("nRecHits", fmt::sprintf("%s;Reference;Target",toRep), 15, -0.5, 14.5, 15, -0.5, 14.5);

  toRep = "Number of all layers per track (quality #geq loose)";
  hnLayers_ = iBook.book2I("nLayers", fmt::sprintf("%s;Reference;Target",toRep), 15, -0.5, 14.5, 15, -0.5, 14.5);

  toRep = "Track (quality #geq loose) #chi^{2}/ndof";
  hchi2_ = iBook.book2I("nChi2ndof", fmt::sprintf("%s;Reference;Target",toRep), 40, 0., 20., 40, 0., 20.);

  toRep = "Track (quality #geq loose) charge";
  hCharge_ = iBook.book2I("charge",fmt::sprintf("%s;Reference;Target",toRep),3, -1.5, 1.5, 3, -1.5, 1.5);

  hpt_ = iBook.book2I("pt", "Track (quality #geq loose) p_{T} [GeV];Reference;Target", 200, 0., 200., 200, 0., 200.);
  hptLogLog_ = make2DIfLog(iBook, true, true, "ptLogLog", "Track (quality #geq loose) p_{T} [GeV];Reference;Target", 200, log10(0.5), log10(200.), 200, log10(0.5), log10(200.));
  heta_ = iBook.book2I("eta", "Track (quality #geq loose) #eta;Reference;Target", 30, -3., 3., 30, -3., 3.);
  hphi_ = iBook.book2I("phi", "Track (quality #geq loose) #phi;Reference;Target", 30, -M_PI, M_PI, 30, -M_PI, M_PI);
  hz_ = iBook.book2I("z", "Track (quality #geq loose) z [cm];Reference;Target", 30, -30., 30., 30, -30., 30.);
  htip_ = iBook.book2I("tip", "Track (quality #geq loose) TIP [cm];Reference;Target", 100, -0.5, 0.5, 100, -0.5, 0.5);
  //1D difference plots
  hptdiffMatched_ = iBook.book1D("ptdiffmatched", " p_{T} diff [GeV] between matched tracks; #Delta p_{T} [GeV]", 60, -30., 30.);
  hCurvdiffMatched_ = iBook.book1D("curvdiffmatched", "q/p_{T} diff [GeV] between matched tracks; #Delta q/p_{T} [GeV]", 60, -30., 30.);
  hetadiffMatched_ = iBook.book1D("etadiffmatched", " #eta diff between matched tracks; #Delta #eta", 160, -0.04 ,0.04);
  hphidiffMatched_ = iBook.book1D("phidiffmatched", " #phi diff between matched tracks; #Delta #phi",  160, -0.04 ,0.04);
  hzdiffMatched_ = iBook.book1D("zdiffmatched", " z diff between matched tracks; #Delta z [cm]", 300, -1.5, 1.5);
  htipdiffMatched_ = iBook.book1D("tipdiffmatched", " TIP diff between matched tracks; #Delta TIP [cm]", 300, -1.5, 1.5);
  //2D plots for eff
  hpt_eta_tkAllRef_ = iBook.book2I("ptetatrkAllReference", "Track (quality #geq loose) on Reference; #eta; p_{T} [GeV];", 30, -M_PI, M_PI, 200, 0., 200.);
  hpt_eta_tkAllRefMatched_ = iBook.book2I("ptetatrkAllReferencematched", "Track (quality #geq loose) on Reference matched to Target track; #eta; p_{T} [GeV];", 30, -M_PI, M_PI, 200, 0., 200.);

  hphi_z_tkAllRef_ = iBook.book2I("phiztrkAllReference", "Track (quality #geq loose) on Reference; #phi; z [cm];",  30, -M_PI, M_PI, 30, -30., 30.);
  hphi_z_tkAllRefMatched_ = iBook.book2I("phiztrkAllReferencematched", "Track (quality #geq loose) on Reference; #phi; z [cm];", 30, -M_PI, M_PI, 30, -30., 30.);

}

template<typename T>
void SiPixelCompareTracks<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelTrackSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelTrackReferenceSoA", edm::InputTag("pixelTracksAlpakaSerial"));
  desc.add<edm::InputTag>("pixelTrackTargetSoA", edm::InputTag("pixelTracksAlpaka"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelTrackCompareDeviceVSHost");
  desc.add<bool>("useQualityCut", true);
  desc.add<std::string>("minQuality", "loose");
  desc.add<double>("deltaR2cut", 0.04);
  descriptions.addWithDefaultLabel(desc);
}

// TODO: change module names to SiPixel*CompareTracksSoA when CUDA code is removed

using SiPixelPhase1CompareTracks = SiPixelCompareTracks<pixelTopology::Phase1>;
using SiPixelPhase2CompareTracks = SiPixelCompareTracks<pixelTopology::Phase2>;
using SiPixelHIonPhase1CompareTracks = SiPixelCompareTracks<pixelTopology::HIonPhase1>;

DEFINE_FWK_MODULE(SiPixelPhase1CompareTracks);
DEFINE_FWK_MODULE(SiPixelPhase2CompareTracks);
DEFINE_FWK_MODULE(SiPixelHIonPhase1CompareTracks);

