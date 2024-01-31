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

template <typename T>
class SiPixelCompareTrackSoAAlpaka : public DQMEDAnalyzer {
public:
  using PixelTrackSoA = TracksHost<T>;

  explicit SiPixelCompareTrackSoAAlpaka(const edm::ParameterSet&);
  ~SiPixelCompareTrackSoAAlpaka() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<PixelTrackSoA> tokenSoATrackHost_;
  const edm::EDGetTokenT<PixelTrackSoA> tokenSoATrackDevice_;
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
  MonitorElement* hpt_eta_tkAllHost_;
  MonitorElement* hpt_eta_tkAllHostMatched_;
  MonitorElement* hphi_z_tkAllHost_;
  MonitorElement* hphi_z_tkAllHostMatched_;
};

//
// constructors
//

template <typename T>
SiPixelCompareTrackSoAAlpaka<T>::SiPixelCompareTrackSoAAlpaka(const edm::ParameterSet& iConfig)
    : tokenSoATrackHost_(consumes<PixelTrackSoA>(iConfig.getParameter<edm::InputTag>("pixelTrackSrcHost"))),
      tokenSoATrackDevice_(consumes<PixelTrackSoA>(iConfig.getParameter<edm::InputTag>("pixelTrackSrcDevice"))),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")),
      useQualityCut_(iConfig.getParameter<bool>("useQualityCut")),
      minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
      dr2cut_(iConfig.getParameter<double>("deltaR2cut")) {}

//
// -- Analyze
//
template <typename T>
void SiPixelCompareTrackSoAAlpaka<T>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using helper = TracksUtilities<T>;
  const auto& tsoaHandleHost = iEvent.getHandle(tokenSoATrackHost_);
  const auto& tsoaHandleDevice = iEvent.getHandle(tokenSoATrackDevice_);
  if (not tsoaHandleHost or not tsoaHandleDevice) {
    edm::LogWarning out("SiPixelCompareTrackSoAAlpaka");
    if (not tsoaHandleHost) {
      out << "reference (cpu) tracks not found; ";
    }
    if (not tsoaHandleDevice) {
      out << "target (gpu) tracks not found; ";
    }
    out << "the comparison will not run.";
    return;
  }

  auto const& tsoaHost = *tsoaHandleHost;
  auto const& tsoaDevice = *tsoaHandleDevice;
  auto maxTracksHost = tsoaHost.view().metadata().size();      //this should be same for both?
  auto maxTracksDevice = tsoaDevice.view().metadata().size();  //this should be same for both?
  auto const* qualityHost = tsoaHost.view().quality();
  auto const* qualityDevice = tsoaDevice.view().quality();
  int32_t nTracksHost = 0;
  int32_t nTracksDevice = 0;
  int32_t nLooseAndAboveTracksHost = 0;
  int32_t nLooseAndAboveTracksHost_matchedDevice = 0;
  int32_t nLooseAndAboveTracksDevice = 0;

  //Loop over Device tracks and store the indices of the loose tracks. Whats happens if useQualityCut_ is false?
  std::vector<int32_t> looseTrkidxDevice;
  for (int32_t jt = 0; jt < maxTracksDevice; ++jt) {
    if (helper::nHits(tsoaDevice.view(), jt) == 0)
      break;  // this is a guard
    if (!(tsoaDevice.view()[jt].pt() > 0.))
      continue;
    nTracksDevice++;
    if (useQualityCut_ && qualityDevice[jt] < minQuality_)
      continue;
    nLooseAndAboveTracksDevice++;
    looseTrkidxDevice.emplace_back(jt);
  }

  //Now loop over Host tracks//nested loop for loose gPU tracks
  for (int32_t it = 0; it < maxTracksHost; ++it) {
    int nHitsHost = helper::nHits(tsoaHost.view(), it);

    if (nHitsHost == 0)
      break;  // this is a guard

    float ptHost = tsoaHost.view()[it].pt();
    float etaHost = tsoaHost.view()[it].eta();
    float phiHost = reco::phi(tsoaHost.view(), it);
    float zipHost = reco::zip(tsoaHost.view(), it);
    float tipHost = reco::tip(tsoaHost.view(), it);

    if (!(ptHost > 0.))
      continue;
    nTracksHost++;
    if (useQualityCut_ && qualityHost[it] < minQuality_)
      continue;
    nLooseAndAboveTracksHost++;
    //Now loop over loose Device trk and find the closest in DeltaR//do we need pt cut?
    const int32_t notFound = -1;
    int32_t closestTkidx = notFound;
    float mindr2 = dr2cut_;

    for (auto gid : looseTrkidxDevice) {
      float etaDevice = tsoaDevice.view()[gid].eta();
      float phiDevice = reco::phi(tsoaDevice.view(), gid);
      float dr2 = reco::deltaR2(etaHost, phiHost, etaDevice, phiDevice);
      if (dr2 > dr2cut_)
        continue;  // this is arbitrary
      if (mindr2 > dr2) {
        mindr2 = dr2;
        closestTkidx = gid;
      }
    }

    hpt_eta_tkAllHost_->Fill(etaHost, ptHost);  //all Host tk
    hphi_z_tkAllHost_->Fill(phiHost, zipHost);
    if (closestTkidx == notFound)
      continue;
    nLooseAndAboveTracksHost_matchedDevice++;

    hchi2_->Fill(tsoaHost.view()[it].chi2(), tsoaDevice.view()[closestTkidx].chi2());
    hCharge_->Fill(reco::charge(tsoaHost.view(), it), reco::charge(tsoaDevice.view(), closestTkidx));
    hnHits_->Fill(helper::nHits(tsoaHost.view(), it), helper::nHits(tsoaDevice.view(), closestTkidx));
    hnLayers_->Fill(tsoaHost.view()[it].nLayers(), tsoaDevice.view()[closestTkidx].nLayers());
    hpt_->Fill(tsoaHost.view()[it].pt(), tsoaDevice.view()[closestTkidx].pt());
    hptLogLog_->Fill(tsoaHost.view()[it].pt(), tsoaDevice.view()[closestTkidx].pt());
    heta_->Fill(etaHost, tsoaDevice.view()[closestTkidx].eta());
    hphi_->Fill(phiHost, reco::phi(tsoaDevice.view(), closestTkidx));
    hz_->Fill(zipHost, reco::zip(tsoaDevice.view(), closestTkidx));
    htip_->Fill(tipHost, reco::tip(tsoaDevice.view(), closestTkidx));
    hptdiffMatched_->Fill(ptHost - tsoaDevice.view()[closestTkidx].pt());
    hCurvdiffMatched_->Fill((reco::charge(tsoaHost.view(), it) / tsoaHost.view()[it].pt()) -
                            (reco::charge(tsoaDevice.view(), closestTkidx) / tsoaDevice.view()[closestTkidx].pt()));
    hetadiffMatched_->Fill(etaHost - tsoaDevice.view()[closestTkidx].eta());
    hphidiffMatched_->Fill(reco::deltaPhi(phiHost, reco::phi(tsoaDevice.view(), closestTkidx)));
    hzdiffMatched_->Fill(zipHost - reco::zip(tsoaDevice.view(), closestTkidx));
    htipdiffMatched_->Fill(tipHost - reco::tip(tsoaDevice.view(), closestTkidx));
    hpt_eta_tkAllHostMatched_->Fill(etaHost, tsoaHost.view()[it].pt());  //matched to gpu
    hphi_z_tkAllHostMatched_->Fill(etaHost, zipHost);
  }
  hnTracks_->Fill(nTracksHost, nTracksDevice);
  hnLooseAndAboveTracks_->Fill(nLooseAndAboveTracksHost, nLooseAndAboveTracksDevice);
  hnLooseAndAboveTracks_matched_->Fill(nLooseAndAboveTracksHost, nLooseAndAboveTracksHost_matchedDevice);
}

//
// -- Book Histograms
//
template <typename T>
void SiPixelCompareTrackSoAAlpaka<T>::bookHistograms(DQMStore::IBooker& iBook,
                                                     edm::Run const& iRun,
                                                     edm::EventSetup const& iSetup) {
  iBook.cd();
  iBook.setCurrentFolder(topFolderName_);

  // clang-format off
  std::string toRep = "Number of tracks";
  // FIXME: all the 2D correlation plots are quite heavy in terms of memory consumption, so a as soon as DQM supports THnSparse
  // these should be moved to a less resource consuming format
  hnTracks_ = iBook.book2I("nTracks", fmt::format("{} per event; Host; Device",toRep), 501, -0.5, 500.5, 501, -0.5, 500.5);
  hnLooseAndAboveTracks_ = iBook.book2I("nLooseAndAboveTracks", fmt::format("{} (quality #geq loose) per event; Host; Device",toRep), 501, -0.5, 500.5, 501, -0.5, 500.5);
  hnLooseAndAboveTracks_matched_ = iBook.book2I("nLooseAndAboveTracks_matched", fmt::format("{} (quality #geq loose) per event; Host; Device",toRep), 501, -0.5, 500.5, 501, -0.5, 500.5);

  toRep = "Number of all RecHits per track (quality #geq loose)";
  hnHits_ = iBook.book2I("nRecHits", fmt::format("{};Host;Device",toRep), 15, -0.5, 14.5, 15, -0.5, 14.5);

  toRep = "Number of all layers per track (quality #geq loose)";
  hnLayers_ = iBook.book2I("nLayers", fmt::format("{};Host;Device",toRep), 15, -0.5, 14.5, 15, -0.5, 14.5);

  toRep = "Track (quality #geq loose) #chi^{2}/ndof";
  hchi2_ = iBook.book2I("nChi2ndof", fmt::format("{};Host;Device",toRep), 40, 0., 20., 40, 0., 20.);

  toRep = "Track (quality #geq loose) charge";
  hCharge_ = iBook.book2I("charge",fmt::format("{};Host;Device",toRep),3, -1.5, 1.5, 3, -1.5, 1.5);

  hpt_ = iBook.book2I("pt", "Track (quality #geq loose) p_{T} [GeV];Host;Device", 200, 0., 200., 200, 0., 200.);
  hptLogLog_ = make2DIfLog(iBook, true, true, "ptLogLog", "Track (quality #geq loose) p_{T} [GeV];Host;Device", 200, log10(0.5), log10(200.), 200, log10(0.5), log10(200.));
  heta_ = iBook.book2I("eta", "Track (quality #geq loose) #eta;Host;Device", 30, -3., 3., 30, -3., 3.);
  hphi_ = iBook.book2I("phi", "Track (quality #geq loose) #phi;Host;Device", 30, -M_PI, M_PI, 30, -M_PI, M_PI);
  hz_ = iBook.book2I("z", "Track (quality #geq loose) z [cm];Host;Device", 30, -30., 30., 30, -30., 30.);
  htip_ = iBook.book2I("tip", "Track (quality #geq loose) TIP [cm];Host;Device", 100, -0.5, 0.5, 100, -0.5, 0.5);
  //1D difference plots
  hptdiffMatched_ = iBook.book1D("ptdiffmatched", " p_{T} diff [GeV] between matched tracks; #Delta p_{T} [GeV]", 60, -30., 30.);
  hCurvdiffMatched_ = iBook.book1D("curvdiffmatched", "q/p_{T} diff [GeV] between matched tracks; #Delta q/p_{T} [GeV]", 60, -30., 30.);
  hetadiffMatched_ = iBook.book1D("etadiffmatched", " #eta diff between matched tracks; #Delta #eta", 160, -0.04 ,0.04);
  hphidiffMatched_ = iBook.book1D("phidiffmatched", " #phi diff between matched tracks; #Delta #phi",  160, -0.04 ,0.04);
  hzdiffMatched_ = iBook.book1D("zdiffmatched", " z diff between matched tracks; #Delta z [cm]", 300, -1.5, 1.5);
  htipdiffMatched_ = iBook.book1D("tipdiffmatched", " TIP diff between matched tracks; #Delta TIP [cm]", 300, -1.5, 1.5);
  //2D plots for eff
  hpt_eta_tkAllHost_ = iBook.book2I("ptetatrkAllHost", "Track (quality #geq loose) on Host; #eta; p_{T} [GeV];", 30, -M_PI, M_PI, 200, 0., 200.);
  hpt_eta_tkAllHostMatched_ = iBook.book2I("ptetatrkAllHostmatched", "Track (quality #geq loose) on Host matched to Device track; #eta; p_{T} [GeV];", 30, -M_PI, M_PI, 200, 0., 200.);

  hphi_z_tkAllHost_ = iBook.book2I("phiztrkAllHost", "Track (quality #geq loose) on Host; #phi; z [cm];",  30, -M_PI, M_PI, 30, -30., 30.);
  hphi_z_tkAllHostMatched_ = iBook.book2I("phiztrkAllHostmatched", "Track (quality #geq loose) on Host; #phi; z [cm];", 30, -M_PI, M_PI, 30, -30., 30.);

}

template<typename T>
void SiPixelCompareTrackSoAAlpaka<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelTrackSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelTrackSrcHost", edm::InputTag("pixelTracksAlpakaSerial"));
  desc.add<edm::InputTag>("pixelTrackSrcDevice", edm::InputTag("pixelTracksAlpaka"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelTrackCompareDeviceVSHost");
  desc.add<bool>("useQualityCut", true);
  desc.add<std::string>("minQuality", "loose");
  desc.add<double>("deltaR2cut", 0.04);
  descriptions.addWithDefaultLabel(desc);
}

using SiPixelPhase1CompareTrackSoAAlpaka = SiPixelCompareTrackSoAAlpaka<pixelTopology::Phase1>;
using SiPixelPhase2CompareTrackSoAAlpaka = SiPixelCompareTrackSoAAlpaka<pixelTopology::Phase2>;
using SiPixelHIonPhase1CompareTrackSoAAlpaka = SiPixelCompareTrackSoAAlpaka<pixelTopology::HIonPhase1>;

DEFINE_FWK_MODULE(SiPixelPhase1CompareTrackSoAAlpaka);
DEFINE_FWK_MODULE(SiPixelPhase2CompareTrackSoAAlpaka);
DEFINE_FWK_MODULE(SiPixelHIonPhase1CompareTrackSoAAlpaka);
