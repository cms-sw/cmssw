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

class SiPixelPhase1MonitorTrackSoA : public DQMEDAnalyzer {
public:
  explicit SiPixelPhase1MonitorTrackSoA(const edm::ParameterSet&);
  ~SiPixelPhase1MonitorTrackSoA() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<PixelTrackHeterogeneous> tokenSoATrack_;
  std::string topFolderName_;
  bool useQualityCut_;
  pixelTrack::Quality minQuality_;
  MonitorElement* hnTracks;
  MonitorElement* hnHits;
  MonitorElement* hchi2;
  MonitorElement* hpt;
  MonitorElement* heta;
  MonitorElement* hphi;
  MonitorElement* hz;
  MonitorElement* htip;
};

//
// constructors
//

SiPixelPhase1MonitorTrackSoA::SiPixelPhase1MonitorTrackSoA(const edm::ParameterSet& iConfig) {
  tokenSoATrack_ = consumes<PixelTrackHeterogeneous>(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"));
  topFolderName_ = iConfig.getParameter<std::string>("TopFolderName");  //"SiPixelHeterogeneous/PixelTrackSoA";
  useQualityCut_ = iConfig.getParameter<bool>("useQualityCut");
  minQuality_ = pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"));
}

SiPixelPhase1MonitorTrackSoA::~SiPixelPhase1MonitorTrackSoA() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// -- Analyze
//
void SiPixelPhase1MonitorTrackSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& tsoa = *iEvent.get(tokenSoATrack_);
  auto maxTracks = tsoa.stride();
  auto const* quality = tsoa.qualityData();
  int32_t nTracks = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    if (nHits == 0)
      break;  // this is a guard
    float pt = tsoa.pt(it);
    if (!(pt > 0.))
      continue;
    if (useQualityCut_ && quality[it] < minQuality_)
      continue;
    float chi2 = tsoa.chi2(it);
    float phi = tsoa.phi(it);
    float zip = tsoa.zip(it);
    float eta = tsoa.eta(it);
    float tip = tsoa.tip(it);
    hchi2->Fill(chi2);
    hnHits->Fill(nHits);
    hpt->Fill(pt);
    heta->Fill(eta);
    hphi->Fill(phi);
    hz->Fill(zip);
    htip->Fill(tip);
    nTracks++;
  }
  hnTracks->Fill(nTracks);
}

//
// -- Book Histograms
//
void SiPixelPhase1MonitorTrackSoA::bookHistograms(DQMStore::IBooker& ibooker,
                                                  edm::Run const& iRun,
                                                  edm::EventSetup const& iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder(topFolderName_);
  hnTracks = ibooker.book1D("nTracks", ";Number of tracks per event;#entries", 1001, -0.5, 1000.5);
  hnHits = ibooker.book1D("nRecHits", ";Number of all RecHits per track;#entries", 15, -0.5, 14.5);
  hchi2 = ibooker.book1D("nChi2ndof", ";Track chi-squared over ndof;#entries", 40, 0., 20.);
  hpt = ibooker.book1D("pt", ";Track p_T;#entries", 200, 0., 200.);
  heta = ibooker.book1D("eta", ";Track #eta;#entries", 30, -3., 3.);
  hphi = ibooker.book1D("phi", ";Track #phi;#entries", 30, -M_PI, M_PI);
  hz = ibooker.book1D("z", ";Track z;#entries", 30, -30., 30.);
  htip = ibooker.book1D("tip", "", 100, -0.5, 0.5);
}

void SiPixelPhase1MonitorTrackSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelTrackSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("pixelTracksSoA"));
  desc.add<std::string>("TopFolderName", "SiPixelHeterogeneous/PixelTrackSoA");
  desc.add<bool>("useQualityCut", false);
  desc.add<std::string>("minQuality", "loose");
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelPhase1MonitorTrackSoA);
