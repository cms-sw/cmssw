// -*- C++ -*-
///bookLayer
// Package:    SiPixelPhase1CompareTrackSoA
// Class:      SiPixelPhase1CompareTrackSoA
//
/**\class SiPixelPhase1CompareTrackSoA SiPixelPhase1CompareTrackSoA.cc 
*/
//
// Author: Suvankar Roy Chowdhury
//
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"
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

class SiPixelPhase1CompareTrackSoA : public DQMEDAnalyzer {
public:
  explicit SiPixelPhase1CompareTrackSoA(const edm::ParameterSet&);
  ~SiPixelPhase1CompareTrackSoA() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<PixelTrackHeterogeneous> tokenSoATrackCPU_;
  const edm::EDGetTokenT<PixelTrackHeterogeneous> tokenSoATrackGPU_;
  const std::string topFolderName_;
  const bool useQualityCut_;
  const pixelTrack::Quality minQuality_;
  const float dr2cut_;
  MonitorElement* hnTracks;
  MonitorElement* hnLooseAndAboveTracks;
  MonitorElement* hnLooseAndAboveTracks_matched;
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

SiPixelPhase1CompareTrackSoA::SiPixelPhase1CompareTrackSoA(const edm::ParameterSet& iConfig) :
  tokenSoATrackCPU_(consumes<PixelTrackHeterogeneous>(iConfig.getParameter<edm::InputTag>("pixelTrackSrcCPU"))),
  tokenSoATrackGPU_(consumes<PixelTrackHeterogeneous>(iConfig.getParameter<edm::InputTag>("pixelTrackSrcGPU"))),
  topFolderName_(iConfig.getParameter<std::string>("TopFolderName")),
  useQualityCut_(iConfig.getParameter<bool>("useQualityCut")),
  minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))), 
  dr2cut_(iConfig.getParameter<double>("deltaR2cut")) {

}

//
// -- Analyze
//
void SiPixelPhase1CompareTrackSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& tsoaHandleCPU = iEvent.getHandle(tokenSoATrackCPU_);
  const auto& tsoaHandleGPU = iEvent.getHandle(tokenSoATrackGPU_);
  if (!tsoaHandleCPU.isValid() || !tsoaHandleGPU) {
    edm::LogWarning("SiPixelPhase1CompareTrackSoA") << "Either GPU or CPU tracks not found! Hence comparison not run!" << std::endl;
    return;
  }

  auto const& tsoaCPU = *((tsoaHandleCPU.product())->get());
  auto const& tsoaGPU = *((tsoaHandleGPU.product())->get());
  auto maxTracksCPU = tsoaCPU.stride();//this should be same for both?
  auto maxTracksGPU = tsoaGPU.stride();//this should be same for both?
  auto const* qualityCPU = tsoaCPU.qualityData();
  auto const* qualityGPU = tsoaGPU.qualityData();
  int32_t nTracksCPU = 0;
  int32_t nTracksGPU = 0;
  int32_t nLooseAndAboveTracksCPU = 0;
  int32_t nLooseAndAboveTracksCPU_matchedGPU = 0;
  int32_t nLooseAndAboveTracksGPU = 0;

  //Loop over GPU tracks and store the indices of the loose tracks. Whats happens if useQualityCut_ is false?
  std::vector<int32_t>  looseTrkidxGPU;
  for (int32_t jt = 0; jt < maxTracksGPU; ++jt) {
    if (tsoaGPU.nHits(jt) == 0)  break;  // this is a guard
    if (!(tsoaGPU.pt(jt) > 0.))  continue;
    nTracksGPU++;
    if (useQualityCut_ && qualityGPU[jt] < minQuality_)
      continue;
    nLooseAndAboveTracksCPU++;
    looseTrkidxGPU.emplace_back(jt);
  }

  //Now loop over CPU tracks//nested loop for loose gPU tracks
  for (int32_t it = 0; it < maxTracksCPU; ++it) {
    if (tsoaCPU.nHits(it) == 0)  break;  // this is a guard
    if (!(tsoaCPU.pt(it) > 0.))  continue;
    nTracksCPU++;
    if (useQualityCut_ && qualityCPU[it] < minQuality_)
      continue;
    nLooseAndAboveTracksCPU++;
    //Now loop over loose GPU trk and find the closest in DeltaR//do we need pt cut?
    int32_t closestTkidx = 99999;
    float mindr2 = 99.;
    float etacpu = tsoaCPU.eta(it);
    float phicpu = tsoaCPU.phi(it);
    for (auto gid :  looseTrkidxGPU) {
      float etagpu = tsoaGPU.eta(gid);
      float phigpu = tsoaGPU.phi(gid);
      float dr2 = reco::deltaR2(etacpu, phicpu, etagpu, phigpu);
      if(dr2 > dr2cut_)  continue; // this is arbitrary
      if(mindr2 > dr2 ) {
	mindr2 = dr2;
	closestTkidx = gid;
      }
    }
    if(closestTkidx == 99999)  continue;
    nLooseAndAboveTracksCPU_matchedGPU++;

    hchi2->Fill(tsoaCPU.chi2(it), tsoaGPU.chi2(closestTkidx));
    hnHits->Fill(tsoaCPU.nHits(it), tsoaGPU.nHits(closestTkidx));
    hnLayers->Fill(tsoaCPU.nLayers(it), tsoaGPU.nLayers(closestTkidx));
    hpt->Fill(tsoaCPU.pt(it), tsoaGPU.pt(closestTkidx));
    heta->Fill(etacpu, tsoaGPU.eta(closestTkidx));
    hphi->Fill(phicpu, tsoaGPU.phi(closestTkidx));
    hz->Fill(tsoaCPU.zip(it), tsoaGPU.zip(closestTkidx));
    htip->Fill(tsoaCPU.tip(it), tsoaGPU.tip(closestTkidx));
  }
  hnTracks->Fill(nTracksCPU, nTracksGPU);
  hnLooseAndAboveTracks->Fill(nLooseAndAboveTracksCPU, nLooseAndAboveTracksGPU);
  hnLooseAndAboveTracks_matched->Fill(nLooseAndAboveTracksCPU, nLooseAndAboveTracksCPU_matchedGPU);
}

//
// -- Book Histograms
//
void SiPixelPhase1CompareTrackSoA::bookHistograms(DQMStore::IBooker& iBook,
                                                  edm::Run const& iRun,
                                                  edm::EventSetup const& iSetup) {
  iBook.cd();
  iBook.setCurrentFolder(topFolderName_);

  // clang-format off
  std::string toRep = "Number of tracks";
  hnTracks = iBook.book2D("nTracks", fmt::sprintf("%s per event; CPU; GPU",toRep), 1001, -0.5, 1000.5, 1001, -0.5, 1000.5);
  hnLooseAndAboveTracks = iBook.book2D("nLooseAndAboveTracks", fmt::sprintf("%s (quality #geq loose) per event; CPU; GPU",toRep), 1001, -0.5, 1000.5, 1001, -0.5, 1000.5);
  hnLooseAndAboveTracks_matched = iBook.book2D("nLooseAndAboveTracks_matched", fmt::sprintf("%s (quality #geq loose) per event; CPU; GPU",toRep), 1001, -0.5, 1000.5, 1001, -0.5, 1000.5);

  toRep = "Number of all RecHits per track (quality #geq loose)";
  hnHits = iBook.book2D("nRecHits", fmt::sprintf("%s;CPU;GPU",toRep), 15, -0.5, 14.5, 15, -0.5, 14.5);

  toRep = "Number of all layers per track (quality #geq loose)";
  hnLayers = iBook.book2D("nLayers", fmt::sprintf("%s;CPU;GPU",toRep), 15, -0.5, 14.5, 15, -0.5, 14.5);

  toRep = "Track (quality #geq loose) #chi^{2}/ndof";
  hchi2 = iBook.book2D("nChi2ndof", fmt::sprintf("%s;CPU;GPU",toRep), 40, 0., 20., 40, 0., 20.);


  hpt = iBook.book2D("pt", "Track (quality #geq loose) p_{T} [GeV];CPU;GPU", 200, 0., 200., 200, 0., 200.);
  heta = iBook.book2D("eta", "Track (quality #geq loose) #eta;CPU;GPU", 30, -3., 3., 30, -3., 3.);
  hphi = iBook.book2D("phi", "Track (quality #geq loose) #phi;CPU;GPU", 30, -M_PI, M_PI, 30, -M_PI, M_PI);
  hz = iBook.book2D("z", "Track (quality #geq loose) z [cm];CPU;GPU", 30, -30., 30., 30, -30., 30.);
  htip = iBook.book2D("tip", "Track (quality #geq loose) TIP [cm];CPU;GPU", 100, -0.5, 0.5, 100, -0.5, 0.5);
}

void SiPixelPhase1CompareTrackSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelTrackSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelTrackSrcCPU", edm::InputTag("pixelTracksSoA@cpu"));
  desc.add<edm::InputTag>("pixelTrackSrcGPU", edm::InputTag("pixelTracksSoA@cuda"));
  desc.add<std::string>("TopFolderName", "SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU/");
  desc.add<bool>("useQualityCut", true);
  desc.add<std::string>("minQuality", "loose");
  desc.add<double>("deltaR2cut", 0.04);
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelPhase1CompareTrackSoA);
