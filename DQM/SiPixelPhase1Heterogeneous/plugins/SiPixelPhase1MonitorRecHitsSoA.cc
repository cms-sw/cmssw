// -*- C++ -*-
///bookLayer
// Package:    SiPixelPhase1MonitorRecHitsSoA
// Class:      SiPixelPhase1MonitorRecHitsSoA
//
/**\class SiPixelPhase1MonitorRecHitsSoA SiPixelPhase1MonitorRecHitsSoA.cc 
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

class SiPixelPhase1MonitorRecHitsSoA : public DQMEDAnalyzer {
public:
  explicit SiPixelPhase1MonitorRecHitsSoA(const edm::ParameterSet&);
  ~SiPixelPhase1MonitorRecHitsSoA() override = default;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::EDGetTokenT<TrackingRecHit2DCPU> tokenSoAHitsCPU_;
  const std::string topFolderName_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
  MonitorElement* hnHits;
  MonitorElement* hBFposZP;
  MonitorElement* hBFposZR;
  MonitorElement* hBposXY;
  MonitorElement* hBposZP;
  MonitorElement* hBcharge;
  MonitorElement* hBsizex;
  MonitorElement* hBsizey;
  MonitorElement* hBposZPL[4];
  MonitorElement* hBchargeL[4];
  MonitorElement* hBsizexL[4];
  MonitorElement* hBsizeyL[4];
  MonitorElement* hFposXY;
  MonitorElement* hFposZP;
  MonitorElement* hFcharge;
  MonitorElement* hFsizex;
  MonitorElement* hFsizey;
  MonitorElement* hFposXYD[2][3];
  MonitorElement* hFchargeD[2][3];
  MonitorElement* hFsizexD[2][3];
  MonitorElement* hFsizeyD[2][3];
};

//
// constructors
//

SiPixelPhase1MonitorRecHitsSoA::SiPixelPhase1MonitorRecHitsSoA(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
      tokenSoAHitsCPU_(consumes<TrackingRecHit2DCPU>(iConfig.getParameter<edm::InputTag>("pixelHitsSrc"))),
      topFolderName_(iConfig.getParameter<std::string>("TopFolderName")) {}
//
// Begin Run
//
void SiPixelPhase1MonitorRecHitsSoA::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
}

//
// -- Analyze
//
void SiPixelPhase1MonitorRecHitsSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& rhsoaHandle = iEvent.getHandle(tokenSoAHitsCPU_);
  if (!rhsoaHandle.isValid()) {
    edm::LogWarning("SiPixelPhase1MonitorRecHitsSoA") << "No RecHits SoA found \n returning!" << std::endl;
    return;
  }
  auto const& rhsoa = *rhsoaHandle;
  const TrackingRecHit2DSOAView* soa2d = rhsoa.view();

  uint32_t nHits_ = soa2d->nHits();
  hnHits->Fill(nHits_);
  auto detIds = tkGeom_->detUnitIds();
  for (uint32_t i = 0; i < nHits_; i++) {
    DetId id = detIds[soa2d->detectorIndex(i)];
    float xG = soa2d->xGlobal(i);
    float yG = soa2d->yGlobal(i);
    float zG = soa2d->zGlobal(i);
    float rG = soa2d->rGlobal(i);
    float fphi = short2phi(soa2d->iphi(i));
    uint32_t charge = soa2d->charge(i);
    int16_t sizeX = std::ceil(float(std::abs(soa2d->clusterSizeX(i)) / 8.));
    int16_t sizeY = std::ceil(float(std::abs(soa2d->clusterSizeY(i)) / 8.));
    hBFposZP->Fill(zG, fphi);
    int16_t ysign = yG >= 0 ? 1 : -1;
    hBFposZR->Fill(zG, rG * ysign);
    switch (id.subdetId()) {
      case PixelSubdetector::PixelBarrel:
        hBposXY->Fill(xG, yG);
        hBposZP->Fill(zG, fphi);
        hBcharge->Fill(charge);
        hBsizex->Fill(sizeX);
        hBsizey->Fill(sizeY);
        hBposZPL[tTopo_->pxbLayer(id) - 1]->Fill(zG, fphi);
        hBchargeL[tTopo_->pxbLayer(id) - 1]->Fill(charge);
        hBsizexL[tTopo_->pxbLayer(id) - 1]->Fill(sizeX);
        hBsizeyL[tTopo_->pxbLayer(id) - 1]->Fill(sizeY);
        break;
      case PixelSubdetector::PixelEndcap:
        hFposXY->Fill(xG, yG);
        hFposZP->Fill(zG, fphi);
        hFcharge->Fill(charge);
        hFsizex->Fill(sizeX);
        hFsizey->Fill(sizeY);
        hFposXYD[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(xG, yG);
        hFchargeD[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(charge);
        hFsizexD[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(sizeX);
        hFsizeyD[tTopo_->pxfSide(id) - 1][tTopo_->pxfDisk(id) - 1]->Fill(sizeY);
        break;
    }
  }
}

//
// -- Book Histograms
//
void SiPixelPhase1MonitorRecHitsSoA::bookHistograms(DQMStore::IBooker& iBook,
                                                    edm::Run const& iRun,
                                                    edm::EventSetup const& iSetup) {
  iBook.cd();
  iBook.setCurrentFolder(topFolderName_);

  // clang-format off
  //Global
  hnHits = iBook.book1D("nHits", "RecHits per event;RecHits;#events", 200, 0, 5000);
  hBFposZP = iBook.book2D("recHitsGlobalPosZP", "RecHits position Global;Z;#phi", 1000, -60, 60, 200,-3.2,3.2);
  hBFposZR = iBook.book2D("recHitsGlobalPosZR", "RecHits position Global;Z;R", 1000, -60, 60, 200,-20,20);
  //Barrel
  hBposXY = iBook.book2D("recHitsBarrelPosXY", "RecHits position Barrel;X;Y", 200, -20, 20, 200,-20,20);
  hBposZP = iBook.book2D("recHitsBarrelPosZP", "RecHits position Barrel;Z;#phi", 300, -30, 30, 200,-3.2,3.2);
  hBcharge = iBook.book1D("recHitsBarrelCharge", "RecHits Charge Barrel;Charge;#events", 250, 0, 100000);
  hBsizex = iBook.book1D("recHitsBarrelSizex", "RecHits SizeX Barrel;SizeX;#events", 50, 0, 50);
  hBsizey = iBook.book1D("recHitsBarrelSizey", "RecHits SizeY Barrel;SizeY;#events", 50, 0, 50);
  //Barrel Layer
  for(unsigned int il=0;il<tkGeom_->numberOfLayers(PixelSubdetector::PixelBarrel);il++){
    hBposZPL[il] = iBook.book2D(Form("recHitsBLay%dPosZP",il+1), Form("RecHits position Barrel Layer%d;Z;#phi",il+1), 300, -30, 30, 200,-3.2,3.2);
    hBchargeL[il] = iBook.book1D(Form("recHitsBLay%dCharge",il+1), Form("RecHits Charge Barrel Layer%d;Charge;#events",il+1), 250, 0, 100000);
    hBsizexL[il] = iBook.book1D(Form("recHitsBLay%dSizex",il+1), Form("RecHits SizeX Barrel Layer%d;SizeX;#events",il+1), 50, 0, 50);
    hBsizeyL[il] = iBook.book1D(Form("recHitsBLay%dSizey",il+1), Form("RecHits SizeY Barrel Layer%d;SizeY;#events",il+1), 50, 0, 50);
  }
  //Endcaps
  hFposXY = iBook.book2D("recHitsEndcapsPosXY", "RecHits position Endcaps;X;Y", 200, -20, 20, 200,-20,-20);
  hFposZP = iBook.book2D("recHitsEndcapsPosZP", "RecHits position Endcaps;Z;#phi", 600, -60, 60, 200,-3.2,3.2);
  hFcharge = iBook.book1D("recHitsEndcapsCharge", "RecHits Charge Endcaps;Charge;#events", 250, 0, 100000);
  hFsizex = iBook.book1D("recHitsEndcapsSizex", "RecHits SizeX Endcaps;SizeX;#events", 50, 0, 50);
  hFsizey = iBook.book1D("recHitsEndcapsSizey", "RecHits SizeY Endcaps;SizeY;#events", 50, 0, 50);
  //Endcaps Disk
  for(int is=0;is<2;is++){
    int sign=is==0? -1:1;
    for(unsigned int id=0;id<tkGeom_->numberOfLayers(PixelSubdetector::PixelEndcap);id++){
      hFposXYD[is][id] = iBook.book2D(Form("recHitsFDisk%+dPosXY",id*sign+sign), Form("RecHits position Endcaps Disk%+d;X;Y",id*sign+sign), 200, -20, 20, 200,-20,20);
      hFchargeD[is][id] = iBook.book1D(Form("recHitsFDisk%+dCharge",id*sign+sign), Form("RecHits Charge Endcaps Disk%+d;Charge;#events",id*sign+sign), 250, 0, 100000);
      hFsizexD[is][id] = iBook.book1D(Form("recHitsFDisk%+dSizex",id*sign+sign), Form("RecHits SizeX Endcaps Disk%+d;SizeX;#events",id*sign+sign), 50, 0, 50);
      hFsizeyD[is][id] = iBook.book1D(Form("recHitsFDisk%+dSizey",id*sign+sign), Form("RecHits SizeY Endcaps Disk%+d;SizeY;#events",id*sign+sign), 50, 0, 50);
    }
  }
}

void SiPixelPhase1MonitorRecHitsSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelRecHitsSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelHitsSrc", edm::InputTag("siPixelRecHitsPreSplittingSoA"));
  desc.add<std::string>("TopFolderName", "SiPixelHeterogeneous/PixelRecHitsSoA");
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelPhase1MonitorRecHitsSoA);
