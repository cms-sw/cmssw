// -*- C++ -*-
// Package:    SiPixelPhase1CompareRecHitsSoA
// Class:      SiPixelPhase1CompareRecHitsSoA
//
/**\class SiPixelPhase1CompareRecHitsSoA SiPixelPhase1CompareRecHitsSoA.cc 
*/
//
// Author: Suvankar Roy Chowdhury, Alessandro Rossi
//
#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class SiPixelPhase1CompareRecHitsSoA : public DQMEDAnalyzer {
public:
  explicit SiPixelPhase1CompareRecHitsSoA(const edm::ParameterSet&);
  ~SiPixelPhase1CompareRecHitsSoA() override = default;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::EDGetTokenT<TrackingRecHit2DCPU> tokenSoAHitsCPU_;
  const edm::EDGetTokenT<TrackingRecHit2DCPU> tokenSoAHitsGPU_;
  const std::string topFolderName_;
  const float mind2cut_;
  static constexpr uint32_t invalidHit_ = std::numeric_limits<uint32_t>::max();
  static constexpr float micron_ = 10000.;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
  MonitorElement* hnHits_;
  MonitorElement* hBchargeL_[4];
  MonitorElement* hBsizexL_[4];
  MonitorElement* hBsizeyL_[4];
  MonitorElement* hBposxL_[4];
  MonitorElement* hBposyL_[4];
  MonitorElement* hFchargeD_[2][3];
  MonitorElement* hFsizexD_[2][3];
  MonitorElement* hFsizeyD_[2][3];
  MonitorElement* hFposxD_[2][3];
  MonitorElement* hFposyD_[2][3];
  //differences
  MonitorElement* hBchargeDiff_;
  MonitorElement* hFchargeDiff_;
  MonitorElement* hBsizeXDiff_;
  MonitorElement* hFsizeXDiff_;
  MonitorElement* hBsizeYDiff_;
  MonitorElement* hFsizeYDiff_;
  MonitorElement* hBposXDiff_;
  MonitorElement* hFposXDiff_;
  MonitorElement* hBposYDiff_;
  MonitorElement* hFposYDiff_;
};
//
// constructors
//

SiPixelPhase1CompareRecHitsSoA::SiPixelPhase1CompareRecHitsSoA(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
      tokenSoAHitsCPU_(consumes<TrackingRecHit2DCPU>(iConfig.getParameter<edm::InputTag>("pixelHitsSrcCPU"))),
      tokenSoAHitsGPU_(consumes<TrackingRecHit2DCPU>(iConfig.getParameter<edm::InputTag>("pixelHitsSrcGPU"))),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")),
      mind2cut_(iConfig.getParameter<double>("minD2cut")) {}
//
// Begin Run
//
void SiPixelPhase1CompareRecHitsSoA::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
}

//
// -- Analyze
//
void SiPixelPhase1CompareRecHitsSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& rhsoaHandleCPU = iEvent.getHandle(tokenSoAHitsCPU_);
  const auto& rhsoaHandleGPU = iEvent.getHandle(tokenSoAHitsGPU_);
  if (not rhsoaHandleCPU or not rhsoaHandleGPU) {
    edm::LogWarning out("SiPixelPhase1CompareRecHitSoA");
    if (not rhsoaHandleCPU) {
      out << "reference (cpu) rechits not found; ";
    }
    if (not rhsoaHandleGPU) {
      out << "target (gpu) rechits not found; ";
    }
    out << "the comparison will not run.";
    return;
  }
  auto const& rhsoaCPU = *rhsoaHandleCPU;
  const TrackingRecHit2DSOAView* soa2dCPU = rhsoaCPU.view();
  auto const& rhsoaGPU = *rhsoaHandleGPU;
  const TrackingRecHit2DSOAView* soa2dGPU = rhsoaGPU.view();

  uint32_t nHitsCPU = soa2dCPU->nHits();
  uint32_t nHitsGPU = soa2dGPU->nHits();
  hnHits_->Fill(nHitsCPU, nHitsGPU);
  auto detIds = tkGeom_->detUnitIds();
  for (uint32_t i = 0; i < nHitsCPU; i++) {
    float minD = mind2cut_;
    uint32_t matchedHit = invalidHit_;
    uint16_t indCPU = soa2dCPU->detectorIndex(i);
    float xLocalCPU = soa2dCPU->xLocal(i);
    float yLocalCPU = soa2dCPU->yLocal(i);
    for (uint32_t j = 0; j < nHitsGPU; j++) {
      if (soa2dGPU->detectorIndex(j) == indCPU) {
        float dx = xLocalCPU - soa2dGPU->xLocal(j);
        float dy = yLocalCPU - soa2dGPU->yLocal(j);
        float distance = dx * dx + dy * dy;
        if (distance < minD) {
          minD = distance;
          matchedHit = j;
        }
      }
    }
    DetId id = detIds[indCPU];
    uint32_t chargeCPU = soa2dCPU->charge(i);
    int16_t sizeXCPU = std::ceil(float(std::abs(soa2dCPU->clusterSizeX(i)) / 8.));
    int16_t sizeYCPU = std::ceil(float(std::abs(soa2dCPU->clusterSizeY(i)) / 8.));
    uint32_t chargeGPU = 0;
    int16_t sizeXGPU = -99;
    int16_t sizeYGPU = -99;
    float xLocalGPU = -999.;
    float yLocalGPU = -999.;
    if (matchedHit != invalidHit_) {
      chargeGPU = soa2dGPU->charge(matchedHit);
      sizeXGPU = std::ceil(float(std::abs(soa2dGPU->clusterSizeX(matchedHit)) / 8.));
      sizeYGPU = std::ceil(float(std::abs(soa2dGPU->clusterSizeY(matchedHit)) / 8.));
      xLocalGPU = soa2dGPU->xLocal(matchedHit);
      yLocalGPU = soa2dGPU->yLocal(matchedHit);
    }
    switch (id.subdetId()) {
      case PixelSubdetector::PixelBarrel:
        hBchargeL_[tTopo_->pxbLayer(id) - 1]->Fill(chargeCPU, chargeGPU);
        hBsizexL_[tTopo_->pxbLayer(id) - 1]->Fill(sizeXCPU, sizeXGPU);
        hBsizeyL_[tTopo_->pxbLayer(id) - 1]->Fill(sizeYCPU, sizeYGPU);
        hBposxL_[tTopo_->pxbLayer(id) - 1]->Fill(xLocalCPU, xLocalGPU);
        hBposyL_[tTopo_->pxbLayer(id) - 1]->Fill(yLocalCPU, yLocalGPU);
        hBchargeDiff_->Fill(chargeCPU - chargeGPU);
        hBsizeXDiff_->Fill(sizeXCPU - sizeXGPU);
        hBsizeYDiff_->Fill(sizeYCPU - sizeYGPU);
        hBposXDiff_->Fill(micron_ * (xLocalCPU - xLocalGPU));
        hBposYDiff_->Fill(micron_ * (yLocalCPU - yLocalGPU));
        break;
      case PixelSubdetector::PixelEndcap:
        hFchargeD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(chargeCPU, chargeGPU);
        hFsizexD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(sizeXCPU, sizeXGPU);
        hFsizeyD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(sizeYCPU, sizeYGPU);
        hFposxD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(xLocalCPU, xLocalGPU);
        hFposyD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(yLocalCPU, yLocalGPU);
        hFchargeDiff_->Fill(chargeCPU - chargeGPU);
        hFsizeXDiff_->Fill(sizeXCPU - sizeXGPU);
        hFsizeYDiff_->Fill(sizeYCPU - sizeYGPU);
        hFposXDiff_->Fill(micron_ * (xLocalCPU - xLocalGPU));
        hFposYDiff_->Fill(micron_ * (yLocalCPU - yLocalGPU));
        break;
    }
  }
}

//
// -- Book Histograms
//
void SiPixelPhase1CompareRecHitsSoA::bookHistograms(DQMStore::IBooker& iBook,
                                                    edm::Run const& iRun,
                                                    edm::EventSetup const& iSetup) {
  iBook.cd();
  iBook.setCurrentFolder(topFolderName_);

  // clang-format off
  //Global
  hnHits_ = iBook.book2I("nHits", "CPUvsGPU RecHits per event;#CPU RecHits;#GPU RecHits", 200, 0, 5000,200, 0, 5000);
  //Barrel Layer
  for(unsigned int il=0;il<tkGeom_->numberOfLayers(PixelSubdetector::PixelBarrel);il++){
    hBchargeL_[il] = iBook.book2I(Form("recHitsBLay%dCharge",il+1), Form("CPUvsGPU RecHits Charge Barrel Layer%d;CPU Charge;GPU Charge",il+1), 250, 0, 100000, 250, 0, 100000);
    hBsizexL_[il] = iBook.book2I(Form("recHitsBLay%dSizex",il+1), Form("CPUvsGPU RecHits SizeX Barrel Layer%d;CPU SizeX;GPU SizeX",il+1), 30, 0, 30, 30, 0, 30);
    hBsizeyL_[il] = iBook.book2I(Form("recHitsBLay%dSizey",il+1), Form("CPUvsGPU RecHits SizeY Barrel Layer%d;CPU SizeY;GPU SizeY",il+1), 30, 0, 30, 30, 0, 30);
    hBposxL_[il] = iBook.book2D(Form("recHitsBLay%dPosx",il+1), Form("CPUvsGPU RecHits x-pos in Barrel Layer%d;CPU pos x;GPU pos x",il+1), 200, -5, 5, 200,-5,5);
    hBposyL_[il] = iBook.book2D(Form("recHitsBLay%dPosy",il+1), Form("CPUvsGPU RecHits y-pos in Barrel Layer%d;CPU pos y;GPU pos y",il+1), 200, -5, 5, 200,-5,5);
  }
  //Endcaps
  //Endcaps Disk
  for(int is=0;is<2;is++){
    int sign=is==0? -1:1;
    for(unsigned int id=0;id<tkGeom_->numberOfLayers(PixelSubdetector::PixelEndcap);id++){
      hFchargeD_[is][id] = iBook.book2I(Form("recHitsFDisk%+dCharge",id*sign+sign), Form("CPUvsGPU RecHits Charge Endcaps Disk%+d;CPU Charge;GPU Charge",id*sign+sign), 250, 0, 100000, 250, 0, 100000);
      hFsizexD_[is][id] = iBook.book2I(Form("recHitsFDisk%+dSizex",id*sign+sign), Form("CPUvsGPU RecHits SizeX Endcaps Disk%+d;CPU SizeX;GPU SizeX",id*sign+sign), 30, 0, 30, 30, 0, 30);
      hFsizeyD_[is][id] = iBook.book2I(Form("recHitsFDisk%+dSizey",id*sign+sign), Form("CPUvsGPU RecHits SizeY Endcaps Disk%+d;CPU SizeY;GPU SizeY",id*sign+sign), 30, 0, 30, 30, 0, 30);
      hFposxD_[is][id] = iBook.book2D(Form("recHitsFDisk%+dPosx",id*sign+sign), Form("CPUvsGPU RecHits x-pos Endcaps Disk%+d;CPU pos x;GPU pos x",id*sign+sign), 200, -5, 5, 200, -5, 5);
      hFposyD_[is][id] = iBook.book2D(Form("recHitsFDisk%+dPosy",id*sign+sign), Form("CPUvsGPU RecHits y-pos Endcaps Disk%+d;CPU pos y;GPU pos y",id*sign+sign), 200, -5, 5, 200, -5, 5);
    }
  }
  //1D differences
  hBchargeDiff_ = iBook.book1D("rechitChargeDiffBpix","Charge differnce of rechits in BPix; rechit charge difference (CPU - GPU)", 101, -50.5, 50.5);
  hFchargeDiff_ = iBook.book1D("rechitChargeDiffFpix","Charge differnce of rechits in FPix; rechit charge difference (CPU - GPU)", 101, -50.5, 50.5);
  hBsizeXDiff_ = iBook.book1D("rechitsizeXDiffBpix","SizeX difference of rechits in BPix; rechit sizex difference (CPU - GPU)", 21, -10.5, 10.5);
  hFsizeXDiff_ = iBook.book1D("rechitsizeXDiffFpix","SizeX difference of rechits in FPix; rechit sizex difference (CPU - GPU)", 21, -10.5, 10.5);
  hBsizeYDiff_ = iBook.book1D("rechitsizeYDiffBpix","SizeY difference of rechits in BPix; rechit sizey difference (CPU - GPU)", 21, -10.5, 10.5);
  hFsizeYDiff_ = iBook.book1D("rechitsizeYDiffFpix","SizeY difference of rechits in FPix; rechit sizey difference (CPU - GPU)", 21, -10.5, 10.5);
  hBposXDiff_ = iBook.book1D("rechitsposXDiffBpix","x-position difference of rechits in BPix; rechit x-pos difference (CPU - GPU)", 1000, -10, 10);
  hFposXDiff_ = iBook.book1D("rechitsposXDiffFpix","x-position difference of rechits in FPix; rechit x-pos difference (CPU - GPU)", 1000, -10, 10);
  hBposYDiff_ = iBook.book1D("rechitsposYDiffBpix","y-position difference of rechits in BPix; rechit y-pos difference (CPU - GPU)", 1000, -10, 10);
  hFposYDiff_ = iBook.book1D("rechitsposYDiffFpix","y-position difference of rechits in FPix; rechit y-pos difference (CPU - GPU)", 1000, -10, 10);
}

void SiPixelPhase1CompareRecHitsSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelRecHitsSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelHitsSrcCPU", edm::InputTag("siPixelRecHitsPreSplittingSoA@cpu"));
  desc.add<edm::InputTag>("pixelHitsSrcGPU", edm::InputTag("siPixelRecHitsPreSplittingSoA@cuda"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelRecHitsCompareGPUvsCPU");
  desc.add<double>("minD2cut", 0.0001);
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelPhase1CompareRecHitsSoA);
