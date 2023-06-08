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
  const edm::EDGetTokenT<HitsOnHost> tokenSoAHitsHost_;    //these two are both on Host but originally they have been
  const edm::EDGetTokenT<HitsOnHost> tokenSoAHitsDevice_;  //produced on Host or on Device
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
      tokenSoAHitsHost_(consumes(iConfig.getParameter<edm::InputTag>("pixelHitsSrcCPU"))),
      tokenSoAHitsDevice_(consumes(iConfig.getParameter<edm::InputTag>("pixelHitsSrcGPU"))),
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
  const auto& rhsoaHandleHost = iEvent.getHandle(tokenSoAHitsHost_);
  const auto& rhsoaHandleDevice = iEvent.getHandle(tokenSoAHitsDevice_);
  if (not rhsoaHandleHost or not rhsoaHandleDevice) {
    edm::LogWarning out("SiPixelCompareRecHitSoA");
    if (not rhsoaHandleHost) {
      out << "reference (Host) rechits not found; ";
    }
    if (not rhsoaHandleDevice) {
      out << "target (Device) rechits not found; ";
    }
    out << "the comparison will not run.";
    return;
  }

  auto const& rhsoaHost = *rhsoaHandleHost;
  auto const& rhsoaDevice = *rhsoaHandleDevice;

  auto const& soa2dHost = rhsoaHost.const_view();
  auto const& soa2dDevice = rhsoaDevice.const_view();

  uint32_t nHitsHost = soa2dHost.nHits();
  uint32_t nHitsDevice = soa2dDevice.nHits();

  hnHits_->Fill(nHitsHost, nHitsDevice);
  auto detIds = tkGeom_->detUnitIds();
  for (uint32_t i = 0; i < nHitsHost; i++) {
    float minD = mind2cut_;
    uint32_t matchedHit = invalidHit_;
    uint16_t indHost = soa2dHost[i].detectorIndex();
    float xLocalHost = soa2dHost[i].xLocal();
    float yLocalHost = soa2dHost[i].yLocal();
    for (uint32_t j = 0; j < nHitsDevice; j++) {
      if (soa2dDevice.detectorIndex(j) == indHost) {
        float dx = xLocalHost - soa2dDevice[j].xLocal();
        float dy = yLocalHost - soa2dDevice[j].yLocal();
        float distance = dx * dx + dy * dy;
        if (distance < minD) {
          minD = distance;
          matchedHit = j;
        }
      }
    }
    DetId id = detIds[indHost];
    uint32_t chargeHost = soa2dHost[i].chargeAndStatus().charge;
    int16_t sizeXHost = std::ceil(float(std::abs(soa2dHost[i].clusterSizeX()) / 8.));
    int16_t sizeYHost = std::ceil(float(std::abs(soa2dHost[i].clusterSizeY()) / 8.));
    uint32_t chargeDevice = 0;
    int16_t sizeXDevice = -99;
    int16_t sizeYDevice = -99;
    float xLocalDevice = -999.;
    float yLocalDevice = -999.;
    if (matchedHit != invalidHit_) {
      chargeDevice = soa2dDevice[matchedHit].chargeAndStatus().charge;
      sizeXDevice = std::ceil(float(std::abs(soa2dDevice[matchedHit].clusterSizeX()) / 8.));
      sizeYDevice = std::ceil(float(std::abs(soa2dDevice[matchedHit].clusterSizeY()) / 8.));
      xLocalDevice = soa2dDevice[matchedHit].xLocal();
      yLocalDevice = soa2dDevice[matchedHit].yLocal();
    }
    switch (id.subdetId()) {
      case PixelSubdetector::PixelBarrel:
        hBchargeL_[tTopo_->pxbLayer(id) - 1]->Fill(chargeHost, chargeDevice);
        hBsizexL_[tTopo_->pxbLayer(id) - 1]->Fill(sizeXHost, sizeXDevice);
        hBsizeyL_[tTopo_->pxbLayer(id) - 1]->Fill(sizeYHost, sizeYDevice);
        hBposxL_[tTopo_->pxbLayer(id) - 1]->Fill(xLocalHost, xLocalDevice);
        hBposyL_[tTopo_->pxbLayer(id) - 1]->Fill(yLocalHost, yLocalDevice);
        hBchargeDiff_->Fill(chargeHost - chargeDevice);
        hBsizeXDiff_->Fill(sizeXHost - sizeXDevice);
        hBsizeYDiff_->Fill(sizeYHost - sizeYDevice);
        hBposXDiff_->Fill(micron_ * (xLocalHost - xLocalDevice));
        hBposYDiff_->Fill(micron_ * (yLocalHost - yLocalDevice));
        break;
      case PixelSubdetector::PixelEndcap:
        hFchargeD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(chargeHost, chargeDevice);
        hFsizexD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(sizeXHost, sizeXDevice);
        hFsizeyD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(sizeYHost, sizeYDevice);
        hFposxD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(xLocalHost, xLocalDevice);
        hFposyD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(yLocalHost, yLocalDevice);
        hFchargeDiff_->Fill(chargeHost - chargeDevice);
        hFsizeXDiff_->Fill(sizeXHost - sizeXDevice);
        hFsizeYDiff_->Fill(sizeYHost - sizeYDevice);
        hFposXDiff_->Fill(micron_ * (xLocalHost - xLocalDevice));
        hFposYDiff_->Fill(micron_ * (yLocalHost - yLocalDevice));
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
  hnHits_ = iBook.book2I("nHits", "HostvsDevice RecHits per event;#Host RecHits;#Device RecHits", 200, 0, 5000,200, 0, 5000);
  //Barrel Layer
  for(unsigned int il=0;il<tkGeom_->numberOfLayers(PixelSubdetector::PixelBarrel);il++){
    hBchargeL_[il] = iBook.book2I(Form("recHitsBLay%dCharge",il+1), Form("HostvsDevice RecHits Charge Barrel Layer%d;Host Charge;Device Charge",il+1), 250, 0, 100000, 250, 0, 100000);
    hBsizexL_[il] = iBook.book2I(Form("recHitsBLay%dSizex",il+1), Form("HostvsDevice RecHits SizeX Barrel Layer%d;Host SizeX;Device SizeX",il+1), 30, 0, 30, 30, 0, 30);
    hBsizeyL_[il] = iBook.book2I(Form("recHitsBLay%dSizey",il+1), Form("HostvsDevice RecHits SizeY Barrel Layer%d;Host SizeY;Device SizeY",il+1), 30, 0, 30, 30, 0, 30);
    hBposxL_[il] = iBook.book2D(Form("recHitsBLay%dPosx",il+1), Form("HostvsDevice RecHits x-pos in Barrel Layer%d;Host pos x;Device pos x",il+1), 200, -5, 5, 200,-5,5);
    hBposyL_[il] = iBook.book2D(Form("recHitsBLay%dPosy",il+1), Form("HostvsDevice RecHits y-pos in Barrel Layer%d;Host pos y;Device pos y",il+1), 200, -5, 5, 200,-5,5);
  }
  //Endcaps
  //Endcaps Disk
  for(int is=0;is<2;is++){
    int sign=is==0? -1:1;
    for(unsigned int id=0;id<tkGeom_->numberOfLayers(PixelSubdetector::PixelEndcap);id++){
      hFchargeD_[is][id] = iBook.book2I(Form("recHitsFDisk%+dCharge",id*sign+sign), Form("HostvsDevice RecHits Charge Endcaps Disk%+d;Host Charge;Device Charge",id*sign+sign), 250, 0, 100000, 250, 0, 100000);
      hFsizexD_[is][id] = iBook.book2I(Form("recHitsFDisk%+dSizex",id*sign+sign), Form("HostvsDevice RecHits SizeX Endcaps Disk%+d;Host SizeX;Device SizeX",id*sign+sign), 30, 0, 30, 30, 0, 30);
      hFsizeyD_[is][id] = iBook.book2I(Form("recHitsFDisk%+dSizey",id*sign+sign), Form("HostvsDevice RecHits SizeY Endcaps Disk%+d;Host SizeY;Device SizeY",id*sign+sign), 30, 0, 30, 30, 0, 30);
      hFposxD_[is][id] = iBook.book2D(Form("recHitsFDisk%+dPosx",id*sign+sign), Form("HostvsDevice RecHits x-pos Endcaps Disk%+d;Host pos x;Device pos x",id*sign+sign), 200, -5, 5, 200, -5, 5);
      hFposyD_[is][id] = iBook.book2D(Form("recHitsFDisk%+dPosy",id*sign+sign), Form("HostvsDevice RecHits y-pos Endcaps Disk%+d;Host pos y;Device pos y",id*sign+sign), 200, -5, 5, 200, -5, 5);
    }
  }
  //1D differences
  hBchargeDiff_ = iBook.book1D("rechitChargeDiffBpix","Charge differnce of rechits in BPix; rechit charge difference (Host - Device)", 101, -50.5, 50.5);
  hFchargeDiff_ = iBook.book1D("rechitChargeDiffFpix","Charge differnce of rechits in FPix; rechit charge difference (Host - Device)", 101, -50.5, 50.5);
  hBsizeXDiff_ = iBook.book1D("rechitsizeXDiffBpix","SizeX difference of rechits in BPix; rechit sizex difference (Host - Device)", 21, -10.5, 10.5);
  hFsizeXDiff_ = iBook.book1D("rechitsizeXDiffFpix","SizeX difference of rechits in FPix; rechit sizex difference (Host - Device)", 21, -10.5, 10.5);
  hBsizeYDiff_ = iBook.book1D("rechitsizeYDiffBpix","SizeY difference of rechits in BPix; rechit sizey difference (Host - Device)", 21, -10.5, 10.5);
  hFsizeYDiff_ = iBook.book1D("rechitsizeYDiffFpix","SizeY difference of rechits in FPix; rechit sizey difference (Host - Device)", 21, -10.5, 10.5);
  hBposXDiff_ = iBook.book1D("rechitsposXDiffBpix","x-position difference of rechits in BPix; rechit x-pos difference (Host - Device)", 1000, -10, 10);
  hFposXDiff_ = iBook.book1D("rechitsposXDiffFpix","x-position difference of rechits in FPix; rechit x-pos difference (Host - Device)", 1000, -10, 10);
  hBposYDiff_ = iBook.book1D("rechitsposYDiffBpix","y-position difference of rechits in BPix; rechit y-pos difference (Host - Device)", 1000, -10, 10);
  hFposYDiff_ = iBook.book1D("rechitsposYDiffFpix","y-position difference of rechits in FPix; rechit y-pos difference (Host - Device)", 1000, -10, 10);
}

template<typename T>
void SiPixelCompareRecHitsSoA<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelRecHitsSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelHitsSrcCPU", edm::InputTag("siPixelRecHitsPreSplittingSoA@cpu"));
  desc.add<edm::InputTag>("pixelHitsSrcGPU", edm::InputTag("siPixelRecHitsPreSplittingSoA@cuda"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelRecHitsCompareDevicevsHost");
  desc.add<double>("minD2cut", 0.0001);
  descriptions.addWithDefaultLabel(desc);
}

using SiPixelPhase1CompareRecHitsSoA = SiPixelCompareRecHitsSoA<pixelTopology::Phase1>;
using SiPixelPhase2CompareRecHitsSoA = SiPixelCompareRecHitsSoA<pixelTopology::Phase2>;

DEFINE_FWK_MODULE(SiPixelPhase1CompareRecHitsSoA);
DEFINE_FWK_MODULE(SiPixelPhase2CompareRecHitsSoA);
