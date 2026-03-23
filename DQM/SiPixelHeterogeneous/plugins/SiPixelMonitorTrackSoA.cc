// -*- C++ -*-
// Package:    DQM/SiPixelHetrogeneous
// Class:      SiPixelMonitorTrackSoA
//
/**\class SiPixelMonitorTrackSoA SiPixelMonitorTrackSoA.cc
*/
//
// Author: Suvankar Roy Chowdhury
//

// system includes
#include <fmt/printf.h>

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class SiPixelMonitorTrackSoA : public DQMEDAnalyzer {
public:
  using PixelTrackHeterogeneous = reco::TracksHost;
  explicit SiPixelMonitorTrackSoA(const edm::ParameterSet&);
  ~SiPixelMonitorTrackSoA() override = default;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<PixelTrackHeterogeneous> tokenSoATrack_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const std::string topFolderName_;

  std::vector<std::string> qualityDefinitions_;
  const TrackerGeometry* tkGeom_ = nullptr;
  bool isPhase2_;

  // Global histogram for the quality distribution of ALL tracks
  MonitorElement* hquality;

  // Struct to hold histograms for a specific quality level
  struct QualityHistograms {
    pixelTrack::Quality minQuality;
    std::string qualityName;

    MonitorElement* hnTracks;
    MonitorElement* hnHits;
    MonitorElement* hnHitsVsPhi;
    MonitorElement* hnHitsVsEta;
    MonitorElement* hnLayers;
    MonitorElement* hnLayersVsPhi;
    MonitorElement* hnLayersVsEta;
    MonitorElement* hchi2;
    MonitorElement* hChi2VsPhi;
    MonitorElement* hChi2VsEta;
    MonitorElement* hpt;
    MonitorElement* hCurvature;
    MonitorElement* heta;
    MonitorElement* hphi;
    MonitorElement* hz;
    MonitorElement* htip;
  };

  std::vector<QualityHistograms> histograms_;
};

SiPixelMonitorTrackSoA::SiPixelMonitorTrackSoA(const edm::ParameterSet& iConfig)
    : tokenSoATrack_{consumes<PixelTrackHeterogeneous>(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"))},
      geomToken_{esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()},
      topFolderName_{iConfig.getParameter<std::string>("topFolderName")},
      qualityDefinitions_{iConfig.getParameter<std::vector<std::string>>("qualityDefinitions")},
      isPhase2_{false} {}

void SiPixelMonitorTrackSoA::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  if ((tkGeom_->isThere(GeomDetEnumerators::P2PXB)) || (tkGeom_->isThere(GeomDetEnumerators::P2PXEC))) {
    isPhase2_ = true;
  }
}

void SiPixelMonitorTrackSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& tsoaHandle = iEvent.getHandle(tokenSoATrack_);
  if (!tsoaHandle.isValid()) {
    edm::LogWarning("SiPixelMonitorTrackSoA") << "No Track SoA found \n returning!" << std::endl;
    return;
  }

  auto const& tsoa = *tsoaHandle.product();
  auto maxTracks = tsoa.view().tracks().metadata().size();
  auto const quality = tsoa.view().tracks().quality();

  // Initialize counters for this event for each configured quality level
  std::vector<int32_t> nTracksPerQuality(histograms_.size(), 0);

  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = reco::nHits(tsoa.const_view().tracks(), it);
    auto nLayers = tsoa.view().tracks()[it].nLayers();
    if (nHits == 0)
      break;  // this is a guard
    float pt = tsoa.view().tracks()[it].pt();
    if (!(pt > 0.))
      continue;

    // fill the global quality for all tracks
    pixelTrack::Quality qual = quality[it];
    hquality->Fill(int(qual));

    // Loop over the configured histogram sets
    for (size_t i = 0; i < histograms_.size(); ++i) {
      auto& histSet = histograms_[i];

      // Check if track meets this specific quality requirement
      if (qual < histSet.minQuality)
        continue;

      nTracksPerQuality[i]++;

      // Retrieve track parameters
      auto track = tsoa.view().tracks()[it];
      float chi2 = track.chi2();
      float phi = track.state()(0);
      float zip = track.state()(4);
      float eta = track.eta();
      float tip = track.state()(1);
      auto charge = reco::charge(tsoa.view().tracks(), it);

      // Fill histograms for this quality set
      histSet.hchi2->Fill(chi2);
      histSet.hChi2VsPhi->Fill(phi, chi2);
      histSet.hChi2VsEta->Fill(eta, chi2);
      histSet.hnHits->Fill(nHits);
      histSet.hnLayers->Fill(nLayers);
      histSet.hnHitsVsPhi->Fill(phi, nHits);
      histSet.hnHitsVsEta->Fill(eta, nHits);
      histSet.hnLayersVsPhi->Fill(phi, nLayers);
      histSet.hnLayersVsEta->Fill(eta, nLayers);
      histSet.hpt->Fill(pt);
      histSet.hCurvature->Fill(charge / pt);
      histSet.heta->Fill(eta);
      histSet.hphi->Fill(phi);
      histSet.hz->Fill(zip);
      histSet.htip->Fill(tip);
    }
  }

  // Fill event-level counts
  for (size_t i = 0; i < histograms_.size(); ++i) {
    histograms_[i].hnTracks->Fill(nTracksPerQuality[i]);
  }
}

void SiPixelMonitorTrackSoA::bookHistograms(DQMStore::IBooker& iBook,
                                            edm::Run const& iRun,
                                            edm::EventSetup const& iSetup) {
  iBook.cd();
  iBook.setCurrentFolder(topFolderName_);

  // Global Quality Histogram
  hquality = iBook.book1D("quality", ";Track Quality;#tracks", 7, -0.5, 6.5);
  uint i = 1;
  for (const auto& q : pixelTrack::qualityName) {
    hquality->setBinLabel(i, q.data(), 1);
    i++;
  }

  // Loop over user defined qualities
  for (const auto& qName : qualityDefinitions_) {
    QualityHistograms histSet;
    histSet.qualityName = qName;
    histSet.minQuality = pixelTrack::qualityByName(qName);

    // Create a sub-folder for this quality to avoid name clashes
    iBook.setCurrentFolder(topFolderName_ + "/" + qName);

    std::string toRep = fmt::format("Number of tracks (quality #geq {})", qName);
    histSet.hnTracks = iBook.book1D("nTracks", fmt::format(";{} per event;#events", toRep), 1001, -0.5, 2001.5);

    // N.B.: we need to book explicitly profiles with the option "" (default) in order to get the error on the mean
    // (see https://root.cern.ch/doc/master/classTProfile.html), otherwise the default in DQMServices/Core/interface/DQMStore.h
    // uses the option "s" (i.e. standard deviation on all y values)

    const double etaMax = isPhase2_ ? 4.1 : 3.;

    toRep = fmt::format("Number of all RecHits per track (quality #geq {})", qName);
    histSet.hnHits = iBook.book1D("nRecHits", fmt::format(";{};#tracks", toRep), 15, -0.5, 14.5);
    histSet.hnHitsVsPhi = iBook.bookProfile("nHitsPerTrackVsPhi",
                                            fmt::format("{} vs track #phi [rad];Track #phi [rad];{}", toRep, toRep),
                                            30,
                                            -M_PI,
                                            M_PI,
                                            0.,
                                            15.,
                                            "");
    histSet.hnHitsVsEta = iBook.bookProfile("nHitsPerTrackVsEta",
                                            fmt::format("{} vs track #eta;Track #eta;{}", toRep, toRep),
                                            30,
                                            -etaMax,
                                            etaMax,
                                            0.,
                                            15.,
                                            "");

    toRep = fmt::format("Number of all layers per track (quality #geq {})", qName);
    histSet.hnLayers = iBook.book1D("nLayers", fmt::format(";{};#tracks", toRep), 15, -0.5, 14.5);
    histSet.hnLayersVsPhi = iBook.bookProfile("nLayersPerTrackVsPhi",
                                              fmt::format("{} vs track #phi [rad];Track #phi [rad];{}", toRep, toRep),
                                              30,
                                              -M_PI,
                                              M_PI,
                                              0.,
                                              15.,
                                              "");
    histSet.hnLayersVsEta = iBook.bookProfile("nLayersPerTrackVsEta",
                                              fmt::format("{} vs track #eta;Track #eta;{}", toRep, toRep),
                                              30,
                                              -etaMax,
                                              etaMax,
                                              0.,
                                              15.,
                                              "");

    toRep = fmt::format("Track (quality #geq {}) #chi^{{2}}/ndof", qName);
    histSet.hchi2 = iBook.book1D("nChi2ndof", fmt::format(";{};#tracks", toRep), 40, 0., 20.);
    histSet.hChi2VsPhi = iBook.bookProfile("nChi2ndofVsPhi",
                                           fmt::format("{} vs track #phi [rad];Track #phi [rad];{}", toRep, toRep),
                                           30,
                                           -M_PI,
                                           M_PI,
                                           0.,
                                           20.,
                                           "");
    histSet.hChi2VsEta = iBook.bookProfile(
        "nChi2ndofVsEta", fmt::format("{} vs track #eta;Track #eta;{}", toRep, toRep), 30, -etaMax, etaMax, 0., 20., "");

    // Standard kinematic plots
    std::string label = fmt::format("Track (quality #geq {})", qName);
    histSet.hpt = iBook.book1D("pt", fmt::format(";{} p_{{T}} [GeV];#tracks", label), 200, 0., 200.);
    histSet.hCurvature =
        iBook.book1D("curvature", fmt::format(";{} q/p_{{T}} [GeV^{{-1}}];#tracks", label), 100, -3., 3.);
    histSet.heta = iBook.book1D("eta", fmt::format(";{} #eta;#tracks", label), 30, -etaMax, etaMax);
    histSet.hphi = iBook.book1D("phi", fmt::format(";{} #phi [rad];#tracks", label), 30, -M_PI, M_PI);
    histSet.hz = iBook.book1D("z", fmt::format(";{} z [cm];#tracks", label), 30, -30., 30.);
    histSet.htip = iBook.book1D("tip", fmt::format(";{} TIP [cm];#tracks", label), 100, -0.5, 0.5);

    histograms_.push_back(histSet);
  }
}

void SiPixelMonitorTrackSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("pixelTracksAlpaka"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelTrackAlpaka");
  // Default configuration now accepts a list.
  desc.add<std::vector<std::string>>("qualityDefinitions", {"loose", "highPurity"});
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelMonitorTrackSoA);
