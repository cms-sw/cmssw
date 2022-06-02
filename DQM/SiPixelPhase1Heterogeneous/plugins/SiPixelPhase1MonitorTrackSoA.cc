// -*- C++ -*-
///bookLayer
// Package:    SiPixelPhase1MonitorTrackSoA
// Class:      SiPixelPhase1MonitorTrackSoA
//
/**\class SiPixelPhase1MonitorTrackSoA SiPixelPhase1MonitorTrackSoA.cc 
*/
//
// Author: Suvankar Roy Chowdhury
//
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
// for string manipulations
#include <fmt/printf.h>

class SiPixelPhase1MonitorTrackSoA : public DQMEDAnalyzer {
public:
  explicit SiPixelPhase1MonitorTrackSoA(const edm::ParameterSet&);
  ~SiPixelPhase1MonitorTrackSoA() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<PixelTrackHeterogeneous> tokenSoATrack_;
  std::string topFolderName_;
  bool useQualityCut_;
  pixelTrack::Quality minQuality_;
  MonitorElement* hnTracks;
  MonitorElement* hnLooseAndAboveTracks;
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
  MonitorElement* heta;
  MonitorElement* hphi;
  MonitorElement* hz;
  MonitorElement* htip;
  MonitorElement* hquality;
};

//
// constructors
//

SiPixelPhase1MonitorTrackSoA::SiPixelPhase1MonitorTrackSoA(const edm::ParameterSet& iConfig) {
  tokenSoATrack_ = consumes<PixelTrackHeterogeneous>(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"));
  topFolderName_ = iConfig.getParameter<std::string>("topFolderName");  //"SiPixelHeterogeneous/PixelTrackSoA";
  useQualityCut_ = iConfig.getParameter<bool>("useQualityCut");
  minQuality_ = pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"));
}

//
// -- Analyze
//
void SiPixelPhase1MonitorTrackSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& tsoaHandle = iEvent.getHandle(tokenSoATrack_);
  if (!tsoaHandle.isValid()) {
    edm::LogWarning("SiPixelPhase1MonitorTrackSoA") << "No Track SoA found \n returning!" << std::endl;
    return;
  }

  auto const& tsoa = *((tsoaHandle.product())->get());
  auto maxTracks = tsoa.stride();
  auto const* quality = tsoa.qualityData();
  int32_t nTracks = 0;
  int32_t nLooseAndAboveTracks = 0;

  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    auto nLayers = tsoa.nLayers(it);
    if (nHits == 0)
      break;  // this is a guard
    float pt = tsoa.pt(it);
    if (!(pt > 0.))
      continue;

    // fill the quality for all tracks
    pixelTrack::Quality qual = tsoa.quality(it);
    hquality->Fill(int(qual));
    nTracks++;

    if (useQualityCut_ && quality[it] < minQuality_)
      continue;

    // fill parameters only for quality >= loose
    float chi2 = tsoa.chi2(it);
    float phi = tsoa.phi(it);
    float zip = tsoa.zip(it);
    float eta = tsoa.eta(it);
    float tip = tsoa.tip(it);

    hchi2->Fill(chi2);
    hChi2VsPhi->Fill(phi, chi2);
    hChi2VsEta->Fill(eta, chi2);
    hnHits->Fill(nHits);
    hnLayers->Fill(nLayers);
    hnHitsVsPhi->Fill(phi, nHits);
    hnHitsVsEta->Fill(eta, nHits);
    hnLayersVsPhi->Fill(phi, nLayers);
    hnLayersVsEta->Fill(eta, nLayers);
    hpt->Fill(pt);
    heta->Fill(eta);
    hphi->Fill(phi);
    hz->Fill(zip);
    htip->Fill(tip);
    nLooseAndAboveTracks++;
  }
  hnTracks->Fill(nTracks);
  hnLooseAndAboveTracks->Fill(nLooseAndAboveTracks);
}

//
// -- Book Histograms
//
void SiPixelPhase1MonitorTrackSoA::bookHistograms(DQMStore::IBooker& iBook,
                                                  edm::Run const& iRun,
                                                  edm::EventSetup const& iSetup) {
  iBook.cd();
  iBook.setCurrentFolder(topFolderName_);

  // clang-format off
  std::string toRep = "Number of tracks";
  hnTracks = iBook.book1D("nTracks", fmt::sprintf(";%s per event;#events",toRep), 1001, -0.5, 1000.5);
  hnLooseAndAboveTracks = iBook.book1D("nLooseAndAboveTracks", fmt::sprintf(";%s (quality #geq loose) per event;#events",toRep), 1001, -0.5, 1000.5);

  toRep = "Number of all RecHits per track (quality #geq loose)";
  hnHits = iBook.book1D("nRecHits", fmt::sprintf(";%s;#tracks",toRep), 15, -0.5, 14.5);
  hnHitsVsPhi = iBook.bookProfile("nHitsPerTrackVsPhi", fmt::sprintf("%s vs track #phi;Track #phi;%s",toRep,toRep), 30, -M_PI, M_PI,0., 15.);
  hnHitsVsEta = iBook.bookProfile("nHitsPerTrackVsEta", fmt::sprintf("%s vs track #eta;Track #eta;%s",toRep,toRep), 30, -3., 3., 0., 15.);

  toRep = "Number of all layers per track (quality #geq loose)";
  hnLayers = iBook.book1D("nLayers", fmt::sprintf(";%s;#tracks",toRep), 15, -0.5, 14.5);
  hnLayersVsPhi = iBook.bookProfile("nLayersPerTrackVsPhi", fmt::sprintf("%s vs track #phi;Track #phi;%s",toRep,toRep), 30, -M_PI, M_PI,0., 15.);
  hnLayersVsEta = iBook.bookProfile("nLayersPerTrackVsEta", fmt::sprintf("%s vs track #eta;Track #eta;%s",toRep,toRep), 30, -3., 3., 0., 15.);

  toRep = "Track (quality #geq loose) #chi^{2}/ndof";
  hchi2 = iBook.book1D("nChi2ndof", fmt::sprintf(";%s;#tracks",toRep), 40, 0., 20.);
  hChi2VsPhi = iBook.bookProfile("nChi2ndofVsPhi", fmt::sprintf("%s vs track #phi;Track #phi;%s",toRep,toRep), 30, -M_PI, M_PI, 0., 20.);
  hChi2VsEta = iBook.bookProfile("nChi2ndofVsEta", fmt::sprintf("%s vs track #eta;Track #eta;%s",toRep,toRep), 30, -3., 3., 0., 20.);
  // clang-format on

  hpt = iBook.book1D("pt", ";Track (quality #geq loose) p_{T} [GeV];#tracks", 200, 0., 200.);
  heta = iBook.book1D("eta", ";Track (quality #geq loose) #eta;#tracks", 30, -3., 3.);
  hphi = iBook.book1D("phi", ";Track (quality #geq loose) #phi;#tracks", 30, -M_PI, M_PI);
  hz = iBook.book1D("z", ";Track (quality #geq loose) z [cm];#tracks", 30, -30., 30.);
  htip = iBook.book1D("tip", ";Track (quality #geq loose) TIP [cm];#tracks", 100, -0.5, 0.5);
  hquality = iBook.book1D("quality", ";Track Quality;#tracks", 7, -0.5, 6.5);
  uint i = 1;
  for (const auto& q : pixelTrack::qualityName) {
    hquality->setBinLabel(i, q, 1);
    i++;
  }
}

void SiPixelPhase1MonitorTrackSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelTrackSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("pixelTracksSoA"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelTrackSoA");
  desc.add<bool>("useQualityCut", true);
  desc.add<std::string>("minQuality", "loose");
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelPhase1MonitorTrackSoA);
