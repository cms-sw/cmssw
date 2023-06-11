// -*- C++ -*-
// Package:    SiPixelCompareRecHitsSoA
// Class:      SiPixelCompareRecHitsSoA
//
/**\class SiPixelCompareRecHitsSoA SiPixelCompareRecHitsSoA.cc
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
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

template <typename T>
class SiPixelCompareRecHitsSoA : public DQMEDAnalyzer {
public:
  using HitSoA = TrackingRecHitSoAView<T>;
  using HitsOnHost = TrackingRecHitSoAHost<T>;

  explicit SiPixelCompareRecHitsSoA(const edm::ParameterSet&);
  ~SiPixelCompareRecHitsSoA() override = default;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::EDGetTokenT<HitsOnHost> tokenSoAHitsCPU_;  //these two are both on CPU but originally they have been
  const edm::EDGetTokenT<HitsOnHost> tokenSoAHitsGPU_;  //produced on CPU or on GPU
  const std::string topFolderName_;
  const float mind2cut_;
  static constexpr uint32_t invalidHit_ = std::numeric_limits<uint32_t>::max();
  static constexpr float micron_ = 10000.;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
  MonitorElement* hnHits_;
  MonitorElement* hBchargeL_[4];  // max 4 barrel hits
  MonitorElement* hBsizexL_[4];
  MonitorElement* hBsizeyL_[4];
  MonitorElement* hBposxL_[4];
  MonitorElement* hBposyL_[4];
  MonitorElement* hFchargeD_[2][12];  // max 12 endcap disks
  MonitorElement* hFsizexD_[2][12];
  MonitorElement* hFsizeyD_[2][12];
  MonitorElement* hFposxD_[2][12];
  MonitorElement* hFposyD_[2][12];
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

template <typename T>
SiPixelCompareRecHitsSoA<T>::SiPixelCompareRecHitsSoA(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
      tokenSoAHitsCPU_(consumes(iConfig.getParameter<edm::InputTag>("pixelHitsSrcCPU"))),
      tokenSoAHitsGPU_(consumes(iConfig.getParameter<edm::InputTag>("pixelHitsSrcGPU"))),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")),
      mind2cut_(iConfig.getParameter<double>("minD2cut")) {}
//
// Begin Run
//
template <typename T>
void SiPixelCompareRecHitsSoA<T>::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
}

//
// -- Analyze
//
template <typename T>
void SiPixelCompareRecHitsSoA<T>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& rhsoaHandleCPU = iEvent.getHandle(tokenSoAHitsCPU_);
  const auto& rhsoaHandleGPU = iEvent.getHandle(tokenSoAHitsGPU_);
  if (not rhsoaHandleCPU or not rhsoaHandleGPU) {
    edm::LogWarning out("SiPixelCompareRecHitSoA");
    if (not rhsoaHandleCPU) {
      out << "reference (CPU) rechits not found; ";
    }
    if (not rhsoaHandleGPU) {
      out << "target (GPU) rechits not found; ";
    }
    out << "the comparison will not run.";
    return;
  }

  auto const& rhsoaCPU = *rhsoaHandleCPU;
  auto const& rhsoaGPU = *rhsoaHandleGPU;

  auto const& soa2dCPU = rhsoaCPU.const_view();
  auto const& soa2dGPU = rhsoaGPU.const_view();

  uint32_t nHitsCPU = soa2dCPU.nHits();
  uint32_t nHitsGPU = soa2dGPU.nHits();

  hnHits_->Fill(nHitsCPU, nHitsGPU);
  auto detIds = tkGeom_->detUnitIds();
  for (uint32_t i = 0; i < nHitsCPU; i++) {
    float minD = mind2cut_;
    uint32_t matchedHit = invalidHit_;
    uint16_t indCPU = soa2dCPU[i].detectorIndex();
    float xLocalCPU = soa2dCPU[i].xLocal();
    float yLocalCPU = soa2dCPU[i].yLocal();
    for (uint32_t j = 0; j < nHitsGPU; j++) {
      if (soa2dGPU.detectorIndex(j) == indCPU) {
        float dx = xLocalCPU - soa2dGPU[j].xLocal();
        float dy = yLocalCPU - soa2dGPU[j].yLocal();
        float distance = dx * dx + dy * dy;
        if (distance < minD) {
          minD = distance;
          matchedHit = j;
        }
      }
    }
    DetId id = detIds[indCPU];
    uint32_t chargeCPU = soa2dCPU[i].chargeAndStatus().charge;
    int16_t sizeXCPU = std::ceil(float(std::abs(soa2dCPU[i].clusterSizeX()) / 8.));
    int16_t sizeYCPU = std::ceil(float(std::abs(soa2dCPU[i].clusterSizeY()) / 8.));
    uint32_t chargeGPU = 0;
    int16_t sizeXGPU = -99;
    int16_t sizeYGPU = -99;
    float xLocalGPU = -999.;
    float yLocalGPU = -999.;
    if (matchedHit != invalidHit_) {
      chargeGPU = soa2dGPU[matchedHit].chargeAndStatus().charge;
      sizeXGPU = std::ceil(float(std::abs(soa2dGPU[matchedHit].clusterSizeX()) / 8.));
      sizeYGPU = std::ceil(float(std::abs(soa2dGPU[matchedHit].clusterSizeY()) / 8.));
      xLocalGPU = soa2dGPU[matchedHit].xLocal();
      yLocalGPU = soa2dGPU[matchedHit].yLocal();
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
template <typename T>
void SiPixelCompareRecHitsSoA<T>::bookHistograms(DQMStore::IBooker& iBook,
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

template<typename T>
void SiPixelCompareRecHitsSoA<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelRecHitsSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelHitsSrcCPU", edm::InputTag("siPixelRecHitsPreSplittingSoA@cpu"));
  desc.add<edm::InputTag>("pixelHitsSrcGPU", edm::InputTag("siPixelRecHitsPreSplittingSoA@cuda"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelRecHitsCompareGPUvsCPU");
  desc.add<double>("minD2cut", 0.0001);
  descriptions.addWithDefaultLabel(desc);
}

using SiPixelPhase1CompareRecHitsSoA = SiPixelCompareRecHitsSoA<pixelTopology::Phase1>;
using SiPixelPhase2CompareRecHitsSoA = SiPixelCompareRecHitsSoA<pixelTopology::Phase2>;

DEFINE_FWK_MODULE(SiPixelPhase1CompareRecHitsSoA);
DEFINE_FWK_MODULE(SiPixelPhase2CompareRecHitsSoA);
