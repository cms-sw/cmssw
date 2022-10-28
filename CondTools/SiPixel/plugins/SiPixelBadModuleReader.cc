// system include files
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/time.h>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// ROOT includes
#include "TCanvas.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH2F.h"

class SiPixelBadModuleReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiPixelBadModuleReader(const edm::ParameterSet &);
  ~SiPixelBadModuleReader() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  const edm::ESGetToken<SiPixelQuality, SiPixelQualityRcd> badModuleToken;
  const edm::ESGetToken<SiPixelQuality, SiPixelQualityFromDbRcd> badModuleFromDBToken;
  const edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> siPixelFedCablingToken;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken;

  const uint32_t printdebug_;
  const std::string whichRcd;

  TH2F *_TH2F_dead_modules_BPIX_lay1;
  TH2F *_TH2F_dead_modules_BPIX_lay2;
  TH2F *_TH2F_dead_modules_BPIX_lay3;
  TH2F *_TH2F_dead_modules_FPIX_minusZ_disk1;
  TH2F *_TH2F_dead_modules_FPIX_minusZ_disk2;
  TH2F *_TH2F_dead_modules_FPIX_plusZ_disk1;
  TH2F *_TH2F_dead_modules_FPIX_plusZ_disk2;
};

SiPixelBadModuleReader::SiPixelBadModuleReader(const edm::ParameterSet &iConfig)
    : badModuleToken(esConsumes()),
      badModuleFromDBToken(esConsumes()),
      siPixelFedCablingToken(esConsumes()),
      tkGeomToken(esConsumes()),
      tkTopoToken(esConsumes()),
      printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 1)),
      whichRcd(iConfig.getUntrackedParameter<std::string>("RcdName")) {
  usesResource(TFileService::kSharedResource);
}
// txtFileName_(iConfig.getUntrackedParameter<std::string>("OutputFile","BadModuleSummary.txt")){}

SiPixelBadModuleReader::~SiPixelBadModuleReader() = default;

void SiPixelBadModuleReader::analyze(const edm::Event &e, const edm::EventSetup &iSetup) {
  const SiPixelQuality *SiPixelBadModule_ = nullptr;
  if (whichRcd == "SiPixelQualityRcd") {
    SiPixelBadModule_ = &iSetup.getData(badModuleToken);
  } else if (whichRcd == "SiPixelQualityFromDbRcd") {
    SiPixelBadModule_ = &iSetup.getData(badModuleFromDBToken);
  } else {
    throw cms::Exception("LogicalError") << "SiPixelBadModuleReader::analyze, unsupported RcdName value " << whichRcd
                                         << ".\n Please check the configuration." << std::endl;
  }

  edm::ESHandle<SiPixelFedCablingMap> map = iSetup.getHandle(siPixelFedCablingToken);
  edm::LogInfo("SiPixelBadModuleReader") << "[SiPixelBadModuleReader::analyze] End Reading SiPixelBadModule"
                                         << std::endl;

  const TrackerGeometry *geom = &iSetup.getData(tkGeomToken);
  const TrackerTopology &ttopo = iSetup.getData(tkTopoToken);

  edm::Service<TFileService> fs;
  _TH2F_dead_modules_BPIX_lay1 =
      fs->make<TH2F>("dead_modules_BPIX_lay1", "Dead modules in BPIX Layer 1", 112, -28., 28., 100, -3.2, 3.2);
  _TH2F_dead_modules_BPIX_lay2 =
      fs->make<TH2F>("dead_modules_BPIX_lay2", "Dead modules in BPIX Layer 2", 112, -28., 28., 100, -3.2, 3.2);
  _TH2F_dead_modules_BPIX_lay3 =
      fs->make<TH2F>("dead_modules_BPIX_lay3", "Dead modules in BPIX Layer 3", 112, -28., 28., 100, -3.2, 3.2);
  _TH2F_dead_modules_FPIX_minusZ_disk1 =
      fs->make<TH2F>("dead_modules_minusZ_disk1", "Dead modules in FPIX minus Z disk 1", 80, -18., 18., 80, -18., 18.);
  _TH2F_dead_modules_FPIX_minusZ_disk2 =
      fs->make<TH2F>("dead_modules_minusZ_disk2", "Dead modules in FPIX minus Z disk 2", 80, -18., 18., 80, -18., 18.);
  _TH2F_dead_modules_FPIX_plusZ_disk1 =
      fs->make<TH2F>("dead_modules_plusZ_disk1", "Dead modules in FPIX plus Z disk 1", 80, -18., 18., 80, -18., 18.);
  _TH2F_dead_modules_FPIX_plusZ_disk2 =
      fs->make<TH2F>("dead_modules_plusZ_disk2", "Dead modules in BPIX plus Z disk 2", 80, -18, 18., 80, -18., 18.);

  gStyle->SetPalette(1);

  std::vector<SiPixelQuality::disabledModuleType> disabledModules = SiPixelBadModule_->getBadComponentList();

  if (printdebug_) {
    std::ofstream debugout("BadModuleDebug.txt");
    debugout << "Values stored in DB, in human readable form: " << std::endl;
    for (size_t id = 0; id < disabledModules.size(); id++) {
      SiPixelQuality::disabledModuleType badmodule = disabledModules[id];

      //////////////////////////////////////
      //  errortype "whole" = int 0 in DB //
      //  errortype "tbmA" = int 1 in DB  //
      //  errortype "tbmB" = int 2 in DB  //
      //////////////////////////////////////

      std::string errorstring;

      if (badmodule.errorType == 0)
        errorstring = "whole";
      else if (badmodule.errorType == 1)
        errorstring = "tbmA";
      else if (badmodule.errorType == 2)
        errorstring = "tbmB";
      else if (badmodule.errorType == 3)
        errorstring = "none";

      debugout << " " << std::endl;
      debugout << " " << std::endl;  //to make the reading easier
      debugout << "DetID: " << badmodule.DetID << " and this has an error type of '" << errorstring << "'" << std::endl;
      debugout << "The bad ROCs are: " << std::endl;
      for (unsigned short n = 0; n < 16; n++) {
        unsigned short mask = 1 << n;  // 1 << n = 2^{n} using bitwise shift
        if (badmodule.BadRocs & mask)
          debugout << n << ", ";
      }
      debugout << std::endl;
      debugout << ttopo.print(badmodule.DetID) << std::endl;
      const auto &plane = geom->idToDet(badmodule.DetID)->surface();
      debugout << "phiSpan " << plane.phiSpan().first << "," << plane.phiSpan().second << std::endl;
      debugout << "rSpan " << plane.rSpan().first << "," << plane.rSpan().second << std::endl;
      debugout << "zSpan " << plane.zSpan().first << "," << plane.zSpan().second << std::endl;
      debugout << " " << std::endl;
      debugout << " " << std::endl;  //to make the reading easier
    }
  }

  int nmodules = 0;
  int nbadmodules = 0;
  int npartialbad = 0;
  for (TrackerGeometry::DetContainer::const_iterator it = geom->dets().begin(); it != geom->dets().end(); it++) {
    if (dynamic_cast<PixelGeomDetUnit const *>((*it)) != nullptr) {
      DetId detId = (*it)->geographicalId();
      uint32_t id = detId();
      nmodules++;

      const GeomDetUnit *geoUnit = geom->idToDetUnit(detId);
      const PixelGeomDetUnit *pixDet = dynamic_cast<const PixelGeomDetUnit *>(geoUnit);
      float detR = pixDet->surface().position().perp();
      float detZ = pixDet->surface().position().z();
      float detPhi = pixDet->surface().position().phi();
      //	  float detEta = -1.*log(tan(atan2(detR,detZ)/2.));
      float detX = detR * cos(detPhi);
      float detY = detR * sin(detPhi);

      //Histograms in "colz":  those with 2 hits are totally fine.  Those with 1 are dead.  Done this way to visualize where ROCs are
      //fill histograms for ALL modules
      if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel) &&
          PixelBarrelName(detId).layerName() == 1) {
        _TH2F_dead_modules_BPIX_lay1->Fill(detZ, detPhi);
        _TH2F_dead_modules_BPIX_lay1->SetOption("colz");
      }

      if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel) &&
          PixelBarrelName(detId).layerName() == 2) {
        _TH2F_dead_modules_BPIX_lay2->Fill(detZ, detPhi);
        _TH2F_dead_modules_BPIX_lay2->SetOption("colz");
      }

      if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel) &&
          PixelBarrelName(detId).layerName() == 3) {
        _TH2F_dead_modules_BPIX_lay3->Fill(detZ, detPhi);
        _TH2F_dead_modules_BPIX_lay3->SetOption("colz");
      }
      if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap) &&
          PixelEndcapName(detId).diskName() == 1 &&
          (PixelEndcapName(detId).halfCylinder() == 2 ||
           PixelEndcapName(detId).halfCylinder() == 1)) {  //mI = 2, mO = 1
        _TH2F_dead_modules_FPIX_minusZ_disk1->Fill(detX, detY);
        _TH2F_dead_modules_FPIX_minusZ_disk1->SetOption("colz");
      }

      if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap) &&
          PixelEndcapName(detId).diskName() == 2 &&
          (PixelEndcapName(detId).halfCylinder() == 2 ||
           PixelEndcapName(detId).halfCylinder() == 1)) {  //mI = 2, mO = 1
        _TH2F_dead_modules_FPIX_minusZ_disk2->Fill(detX, detY);
        _TH2F_dead_modules_FPIX_minusZ_disk2->SetOption("colz");
      }

      if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap) &&
          PixelEndcapName(detId).diskName() == 1 &&
          (PixelEndcapName(detId).halfCylinder() == 3 ||
           PixelEndcapName(detId).halfCylinder() == 4)) {  //p0 = 3, pI = 4
        _TH2F_dead_modules_FPIX_plusZ_disk1->Fill(detX, detY);
        _TH2F_dead_modules_FPIX_plusZ_disk1->SetOption("colz");
      }

      if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap) &&
          PixelEndcapName(detId).diskName() == 2 &&
          (PixelEndcapName(detId).halfCylinder() == 3 ||
           PixelEndcapName(detId).halfCylinder() == 4)) {  //p0 = 3, pI = 4
        _TH2F_dead_modules_FPIX_plusZ_disk2->Fill(detX, detY);
        _TH2F_dead_modules_FPIX_plusZ_disk2->SetOption("colz");
      }

      //fill histograms for when all ROCs are OK
      if (SiPixelBadModule_->IsModuleBad(id) == false && SiPixelBadModule_->getBadRocs(id) == 0) {
        if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel) &&
            PixelBarrelName(detId).layerName() == 1) {
          _TH2F_dead_modules_BPIX_lay1->Fill(detZ, detPhi);
          _TH2F_dead_modules_BPIX_lay1->SetOption("colz");
        }

        if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel) &&
            PixelBarrelName(detId).layerName() == 2) {
          _TH2F_dead_modules_BPIX_lay2->Fill(detZ, detPhi);
          _TH2F_dead_modules_BPIX_lay2->SetOption("colz");
        }

        if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel) &&
            PixelBarrelName(detId).layerName() == 3) {
          _TH2F_dead_modules_BPIX_lay3->Fill(detZ, detPhi);
          _TH2F_dead_modules_BPIX_lay3->SetOption("colz");
        }
        if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap) &&
            PixelEndcapName(detId).diskName() == 1 &&
            (PixelEndcapName(detId).halfCylinder() == 2 ||
             PixelEndcapName(detId).halfCylinder() == 1)) {  //mI = 2, mO = 1
          _TH2F_dead_modules_FPIX_minusZ_disk1->Fill(detX, detY);
          _TH2F_dead_modules_FPIX_minusZ_disk1->SetOption("colz");
        }

        if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap) &&
            PixelEndcapName(detId).diskName() == 2 &&
            (PixelEndcapName(detId).halfCylinder() == 2 ||
             PixelEndcapName(detId).halfCylinder() == 1)) {  //mI = 2, mO = 1
          _TH2F_dead_modules_FPIX_minusZ_disk2->Fill(detX, detY);
          _TH2F_dead_modules_FPIX_minusZ_disk2->SetOption("colz");
        }

        if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap) &&
            PixelEndcapName(detId).diskName() == 1 &&
            (PixelEndcapName(detId).halfCylinder() == 3 ||
             PixelEndcapName(detId).halfCylinder() == 4)) {  //p0 = 3, pI = 4
          _TH2F_dead_modules_FPIX_plusZ_disk1->Fill(detX, detY);
          _TH2F_dead_modules_FPIX_plusZ_disk1->SetOption("colz");
        }

        if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap) &&
            PixelEndcapName(detId).diskName() == 2 &&
            (PixelEndcapName(detId).halfCylinder() == 3 ||
             PixelEndcapName(detId).halfCylinder() == 4)) {  //p0 = 3, pI = 4
          _TH2F_dead_modules_FPIX_plusZ_disk2->Fill(detX, detY);
          _TH2F_dead_modules_FPIX_plusZ_disk2->SetOption("colz");
        }
      }

      //count number of completely dead modules
      if (SiPixelBadModule_->IsModuleBad(id) == true) {
        nbadmodules++;
      }

      //count number of partially dead modules
      if (SiPixelBadModule_->IsModuleBad(id) == false && SiPixelBadModule_->getBadRocs(id) != 0) {
        npartialbad++;
      }
    }
  }

  std::ofstream txtout("BadModuleSummary.txt");
  txtout << "The total number of modules is: " << nmodules << std::endl;
  txtout << "The total number of completely dead modules is: " << nbadmodules << std::endl;
  txtout << "The total number of partially dead modules is: " << npartialbad << std::endl;
}
DEFINE_FWK_MODULE(SiPixelBadModuleReader);
