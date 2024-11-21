// TODO: change file name to SiPixelCompareRecHitsSoA.cc when CUDA code is removed

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// TODO: change class name to SiPixelCompareRecHitsSoA when CUDA code is removed
template <typename T>
class SiPixelCompareRecHits : public DQMEDAnalyzer {
public:
  using HitsSoA = TrackingRecHitHost<T>;

  explicit SiPixelCompareRecHits(const edm::ParameterSet&);
  ~SiPixelCompareRecHits() override = default;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  // analyzeSeparate is templated to accept distinct types of SoAs
  // The default use case is to use rechits from Alpaka reconstructed on CPU and GPU;
  template <typename U, typename V>
  void analyzeSeparate(U tokenRef, V tokenTar, const edm::Event& iEvent);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  // these two are both on Host but originally they have been produced on Host or on Device
  const edm::EDGetTokenT<HitsSoA> tokenSoAHitsReference_;
  const edm::EDGetTokenT<HitsSoA> tokenSoAHitsTarget_;
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
SiPixelCompareRecHits<T>::SiPixelCompareRecHits(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
      tokenSoAHitsReference_(consumes(iConfig.getParameter<edm::InputTag>("pixelHitsReferenceSoA"))),
      tokenSoAHitsTarget_(consumes(iConfig.getParameter<edm::InputTag>("pixelHitsTargetSoA"))),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")),
      mind2cut_(iConfig.getParameter<double>("minD2cut")) {}

//
// Begin Run
//
template <typename T>
void SiPixelCompareRecHits<T>::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
}

template <typename T>
template <typename U, typename V>
void SiPixelCompareRecHits<T>::analyzeSeparate(U tokenRef, V tokenTar, const edm::Event& iEvent) {
  const auto& rhsoaHandleRef = iEvent.getHandle(tokenRef);
  const auto& rhsoaHandleTar = iEvent.getHandle(tokenTar);

  // Exit early if any handle is invalid
  if (!rhsoaHandleRef || !rhsoaHandleTar) {
    edm::LogWarning out("SiPixelCompareRecHits");
    if (!rhsoaHandleRef)
      out << "reference rechits not found; ";
    if (!rhsoaHandleTar)
      out << "target rechits not found; ";
    out << "the comparison will not run.";
    return;
  }

  const auto& rhsoaRef = *rhsoaHandleRef;
  const auto& rhsoaTar = *rhsoaHandleTar;

  auto const& soa2dRef = rhsoaRef.const_view();
  auto const& soa2dTar = rhsoaTar.const_view();

  uint32_t nHitsRef = soa2dRef.metadata().size();
  uint32_t nHitsTar = soa2dTar.metadata().size();

  hnHits_->Fill(nHitsRef, nHitsTar);

  // Map detector indices to target hits for quick access
  std::unordered_map<uint16_t, std::vector<size_t>> detectorIndexMap;
  detectorIndexMap.reserve(nHitsTar);
  for (size_t j = 0; j < nHitsTar; ++j) {
    detectorIndexMap[soa2dTar[j].detectorIndex()].push_back(j);
  }

  auto detIds = tkGeom_->detUnitIds();

  // Loop through reference hits
  for (uint32_t i = 0; i < nHitsRef; i++) {
    float minD = mind2cut_;
    uint32_t matchedHit = invalidHit_;
    uint16_t indRef = soa2dRef[i].detectorIndex();
    float xLocalRef = soa2dRef[i].xLocal();
    float yLocalRef = soa2dRef[i].yLocal();

    // Look up hits in target with matching detector index
    auto it = detectorIndexMap.find(indRef);
    if (it != detectorIndexMap.end()) {
      for (auto j : it->second) {
        float dx = xLocalRef - soa2dTar[j].xLocal();
        float dy = yLocalRef - soa2dTar[j].yLocal();
        float distance = dx * dx + dy * dy;
        if (distance < minD) {
          minD = distance;
          matchedHit = j;
        }
      }
    }

    // Gather reference hit properties
    DetId id = detIds[indRef];
    uint32_t chargeRef = soa2dRef[i].chargeAndStatus().charge;
    int16_t sizeXRef = (soa2dRef[i].clusterSizeX() + 7) / 8;
    int16_t sizeYRef = (soa2dRef[i].clusterSizeY() + 7) / 8;

    // Initialize target hit properties
    uint32_t chargeTar = 0;
    int16_t sizeXTar = -99;
    int16_t sizeYTar = -99;
    float xLocalTar = -999.;
    float yLocalTar = -999.;

    if (matchedHit != invalidHit_) {
      chargeTar = soa2dTar[matchedHit].chargeAndStatus().charge;
      sizeXTar = (soa2dTar[matchedHit].clusterSizeX() + 7) / 8;
      sizeYTar = (soa2dTar[matchedHit].clusterSizeY() + 7) / 8;
      xLocalTar = soa2dTar[matchedHit].xLocal();
      yLocalTar = soa2dTar[matchedHit].yLocal();
    }

    // Populate histograms based on subdetector type
    switch (id.subdetId()) {
      case PixelSubdetector::PixelBarrel:
        hBchargeL_[tTopo_->pxbLayer(id) - 1]->Fill(chargeRef, chargeTar);
        hBsizexL_[tTopo_->pxbLayer(id) - 1]->Fill(sizeXRef, sizeXTar);
        hBsizeyL_[tTopo_->pxbLayer(id) - 1]->Fill(sizeYRef, sizeYTar);
        hBposxL_[tTopo_->pxbLayer(id) - 1]->Fill(xLocalRef, xLocalTar);
        hBposyL_[tTopo_->pxbLayer(id) - 1]->Fill(yLocalRef, yLocalTar);
        hBchargeDiff_->Fill(chargeRef - chargeTar);
        hBsizeXDiff_->Fill(sizeXRef - sizeXTar);
        hBsizeYDiff_->Fill(sizeYRef - sizeYTar);
        hBposXDiff_->Fill(micron_ * (xLocalRef - xLocalTar));
        hBposYDiff_->Fill(micron_ * (yLocalRef - yLocalTar));
        break;
      case PixelSubdetector::PixelEndcap:
        hFchargeD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(chargeRef, chargeTar);
        hFsizexD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(sizeXRef, sizeXTar);
        hFsizeyD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(sizeYRef, sizeYTar);
        hFposxD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(xLocalRef, xLocalTar);
        hFposyD_[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(yLocalRef, yLocalTar);
        hFchargeDiff_->Fill(chargeRef - chargeTar);
        hFsizeXDiff_->Fill(sizeXRef - sizeXTar);
        hFsizeYDiff_->Fill(sizeYRef - sizeYTar);
        hFposXDiff_->Fill(micron_ * (xLocalRef - xLocalTar));
        hFposYDiff_->Fill(micron_ * (yLocalRef - yLocalTar));
        break;
    }
  }
}

//
// -- Analyze
//
template <typename T>
void SiPixelCompareRecHits<T>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // The default use case is to use vertices from Alpaka reconstructed on CPU and GPU;
  // The function is left templated if any other cases need to be added
  analyzeSeparate(tokenSoAHitsReference_, tokenSoAHitsTarget_, iEvent);
}

//
// -- Book Histograms
//
template <typename T>
void SiPixelCompareRecHits<T>::bookHistograms(DQMStore::IBooker& iBook,
                                              edm::Run const& iRun,
                                              edm::EventSetup const& iSetup) {
  iBook.cd();
  iBook.setCurrentFolder(topFolderName_);

  // clang-format off
  //Global
  hnHits_ = iBook.book2I("nHits", "ReferencevsTarget RecHits per event;#Reference RecHits;#Target RecHits", 200, 0, 5000,200, 0, 5000);
  //Barrel Layer
  for(unsigned int il=0;il<tkGeom_->numberOfLayers(PixelSubdetector::PixelBarrel);il++){
    hBchargeL_[il] = iBook.book2I(Form("recHitsBLay%dCharge",il+1), Form("ReferencevsTarget RecHits Charge Barrel Layer%d;Reference Charge;Target Charge",il+1), 250, 0, 100000, 250, 0, 100000);
    hBsizexL_[il] = iBook.book2I(Form("recHitsBLay%dSizex",il+1), Form("ReferencevsTarget RecHits SizeX Barrel Layer%d;Reference SizeX;Target SizeX",il+1), 30, 0, 30, 30, 0, 30);
    hBsizeyL_[il] = iBook.book2I(Form("recHitsBLay%dSizey",il+1), Form("ReferencevsTarget RecHits SizeY Barrel Layer%d;Reference SizeY;Target SizeY",il+1), 30, 0, 30, 30, 0, 30);
    hBposxL_[il] = iBook.book2D(Form("recHitsBLay%dPosx",il+1), Form("ReferencevsTarget RecHits x-pos in Barrel Layer%d;Reference pos x;Target pos x",il+1), 200, -5, 5, 200,-5,5);
    hBposyL_[il] = iBook.book2D(Form("recHitsBLay%dPosy",il+1), Form("ReferencevsTarget RecHits y-pos in Barrel Layer%d;Reference pos y;Target pos y",il+1), 200, -5, 5, 200,-5,5);
  }
  //Endcaps
  //Endcaps Disk
  for(int is=0;is<2;is++){
    int sign=is==0? -1:1;
    for(unsigned int id=0;id<tkGeom_->numberOfLayers(PixelSubdetector::PixelEndcap);id++){
      hFchargeD_[is][id] = iBook.book2I(Form("recHitsFDisk%+dCharge",id*sign+sign), Form("ReferencevsTarget RecHits Charge Endcaps Disk%+d;Reference Charge;Target Charge",id*sign+sign), 250, 0, 100000, 250, 0, 100000);
      hFsizexD_[is][id] = iBook.book2I(Form("recHitsFDisk%+dSizex",id*sign+sign), Form("ReferencevsTarget RecHits SizeX Endcaps Disk%+d;Reference SizeX;Target SizeX",id*sign+sign), 30, 0, 30, 30, 0, 30);
      hFsizeyD_[is][id] = iBook.book2I(Form("recHitsFDisk%+dSizey",id*sign+sign), Form("ReferencevsTarget RecHits SizeY Endcaps Disk%+d;Reference SizeY;Target SizeY",id*sign+sign), 30, 0, 30, 30, 0, 30);
      hFposxD_[is][id] = iBook.book2D(Form("recHitsFDisk%+dPosx",id*sign+sign), Form("ReferencevsTarget RecHits x-pos Endcaps Disk%+d;Reference pos x;Target pos x",id*sign+sign), 200, -5, 5, 200, -5, 5);
      hFposyD_[is][id] = iBook.book2D(Form("recHitsFDisk%+dPosy",id*sign+sign), Form("ReferencevsTarget RecHits y-pos Endcaps Disk%+d;Reference pos y;Target pos y",id*sign+sign), 200, -5, 5, 200, -5, 5);
    }
  }
  //1D differences
  hBchargeDiff_ = iBook.book1D("rechitChargeDiffBpix","Charge differnce of rechits in BPix; rechit charge difference (Reference - Target)", 101, -50.5, 50.5);
  hFchargeDiff_ = iBook.book1D("rechitChargeDiffFpix","Charge differnce of rechits in FPix; rechit charge difference (Reference - Target)", 101, -50.5, 50.5);
  hBsizeXDiff_ = iBook.book1D("rechitsizeXDiffBpix","SizeX difference of rechits in BPix; rechit sizex difference (Reference - Target)", 21, -10.5, 10.5);
  hFsizeXDiff_ = iBook.book1D("rechitsizeXDiffFpix","SizeX difference of rechits in FPix; rechit sizex difference (Reference - Target)", 21, -10.5, 10.5);
  hBsizeYDiff_ = iBook.book1D("rechitsizeYDiffBpix","SizeY difference of rechits in BPix; rechit sizey difference (Reference - Target)", 21, -10.5, 10.5);
  hFsizeYDiff_ = iBook.book1D("rechitsizeYDiffFpix","SizeY difference of rechits in FPix; rechit sizey difference (Reference - Target)", 21, -10.5, 10.5);
  hBposXDiff_ = iBook.book1D("rechitsposXDiffBpix","x-position difference of rechits in BPix; rechit x-pos difference (Reference - Target)", 1000, -10, 10);
  hFposXDiff_ = iBook.book1D("rechitsposXDiffFpix","x-position difference of rechits in FPix; rechit x-pos difference (Reference - Target)", 1000, -10, 10);
  hBposYDiff_ = iBook.book1D("rechitsposYDiffBpix","y-position difference of rechits in BPix; rechit y-pos difference (Reference - Target)", 1000, -10, 10);
  hFposYDiff_ = iBook.book1D("rechitsposYDiffFpix","y-position difference of rechits in FPix; rechit y-pos difference (Reference - Target)", 1000, -10, 10);
}

template<typename T>
void SiPixelCompareRecHits<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelRecHitsSoAAlpaka
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelHitsReferenceSoA", edm::InputTag("siPixelRecHitsPreSplittingAlpakaSerial"));
  desc.add<edm::InputTag>("pixelHitsTargetSoA", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelRecHitsCompareDeviceVSHost");
  desc.add<double>("minD2cut", 0.0001);
  descriptions.addWithDefaultLabel(desc);
}

using SiPixelPhase1CompareRecHits = SiPixelCompareRecHits<pixelTopology::Phase1>;
using SiPixelPhase2CompareRecHits = SiPixelCompareRecHits<pixelTopology::Phase2>;
using SiPixelHIonPhase1CompareRecHits = SiPixelCompareRecHits<pixelTopology::HIonPhase1>;

#include "FWCore/Framework/interface/MakerMacros.h"
// TODO: change module names to SiPixel*CompareRecHitsSoA when CUDA code is removed
DEFINE_FWK_MODULE(SiPixelPhase1CompareRecHits);
DEFINE_FWK_MODULE(SiPixelPhase2CompareRecHits);
DEFINE_FWK_MODULE(SiPixelHIonPhase1CompareRecHits);

