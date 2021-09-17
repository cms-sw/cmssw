// -*- C++ -*-
//
// Package:    SiPixelMonitorCluster
// Class:      SiPixelClusterSource
//
/**\class

 Description: Pixel DQM source for Clusters

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia & Andrew York
//         Created:
//
//
// Updated by: Lukas Wehrli
// for pixel offline DQM
#include "DQM/SiPixelMonitorCluster/interface/SiPixelClusterSource.h"
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// DQM Framework
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQMServices/Core/interface/DQMStore.h"
// Geometry
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
// DataFormats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
//
#include <cstdlib>
#include <string>

using namespace std;
using namespace edm;

SiPixelClusterSource::SiPixelClusterSource(const edm::ParameterSet &iConfig)
    : conf_(iConfig),
      src_(conf_.getParameter<edm::InputTag>("src")),
      digisrc_(conf_.getParameter<edm::InputTag>("digisrc")),
      saveFile(conf_.getUntrackedParameter<bool>("saveFile", false)),
      isPIB(conf_.getUntrackedParameter<bool>("isPIB", false)),
      slowDown(conf_.getUntrackedParameter<bool>("slowDown", false)),
      modOn(conf_.getUntrackedParameter<bool>("modOn", true)),
      twoDimOn(conf_.getUntrackedParameter<bool>("twoDimOn", true)),
      reducedSet(conf_.getUntrackedParameter<bool>("reducedSet", false)),
      ladOn(conf_.getUntrackedParameter<bool>("ladOn", false)),
      layOn(conf_.getUntrackedParameter<bool>("layOn", false)),
      phiOn(conf_.getUntrackedParameter<bool>("phiOn", false)),
      ringOn(conf_.getUntrackedParameter<bool>("ringOn", false)),
      bladeOn(conf_.getUntrackedParameter<bool>("bladeOn", false)),
      diskOn(conf_.getUntrackedParameter<bool>("diskOn", false)),
      smileyOn(conf_.getUntrackedParameter<bool>("smileyOn", false)),
      bigEventSize(conf_.getUntrackedParameter<int>("bigEventSize", 100)),
      isUpgrade(conf_.getUntrackedParameter<bool>("isUpgrade", false)),
      noOfLayers(0),
      noOfDisks(0) {
  LogInfo("PixelDQM") << "SiPixelClusterSource::SiPixelClusterSource: Got DQM BackEnd interface" << endl;

  // set Token(-s)
  srcToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(conf_.getParameter<edm::InputTag>("src"));
  digisrcToken_ = consumes<edm::DetSetVector<PixelDigi>>(conf_.getParameter<edm::InputTag>("digisrc"));

  trackerTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  trackerTopoTokenBeginRun_ = esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>();
  trackerGeomTokenBeginRun_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>();

  firstRun = true;
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
}

SiPixelClusterSource::~SiPixelClusterSource() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  LogInfo("PixelDQM") << "SiPixelClusterSource::~SiPixelClusterSource: Destructor" << endl;

  std::map<uint32_t, SiPixelClusterModule *>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++) {
    delete struct_iter->second;
    struct_iter->second = nullptr;
  }
}

void SiPixelClusterSource::dqmBeginRun(const edm::Run &r, const edm::EventSetup &iSetup) {
  LogInfo("PixelDQM") << " SiPixelClusterSource::beginJob - Initialisation ... " << std::endl;
  LogInfo("PixelDQM") << "Mod/Lad/Lay/Phi " << modOn << "/" << ladOn << "/" << layOn << "/" << phiOn << std::endl;
  LogInfo("PixelDQM") << "Blade/Disk/Ring" << bladeOn << "/" << diskOn << "/" << ringOn << std::endl;
  LogInfo("PixelDQM") << "2DIM IS " << twoDimOn << "\n";
  LogInfo("PixelDQM") << "Smiley (Cluster sizeY vs. Cluster eta) is " << smileyOn << "\n";

  if (firstRun) {
    eventNo = 0;
    lumSec = 0;
    nLumiSecs = 0;
    nBigEvents = 0;
    // Build map
    buildStructure(iSetup);

    firstRun = false;
  }
}

void SiPixelClusterSource::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, const edm::EventSetup &iSetup) {
  bookMEs(iBooker, iSetup);
  // Book occupancy maps in global coordinates for all clusters:
  iBooker.setCurrentFolder(topFolderName_ + "/Clusters/OffTrack");
  // bpix
  std::stringstream ss1, ss2;
  for (int i = 1; i <= noOfLayers; i++) {
    ss1.str(std::string());
    ss1 << "position_siPixelClusters_Layer_" << i;
    ss2.str(std::string());
    ss2 << "Clusters Layer" << i << ";Global Z (cm);Global #phi";
    meClPosLayer.push_back(iBooker.book2D(ss1.str(), ss2.str(), 200, -30., 30., 128, -3.2, 3.2));
  }
  for (int i = 1; i <= noOfDisks; i++) {
    ss1.str(std::string());
    ss1 << "position_siPixelClusters_pz_Disk_" << i;
    ss2.str(std::string());
    ss2 << "Clusters +Z Disk" << i << ";Global X (cm);Global Y (cm)";
    meClPosDiskpz.push_back(iBooker.book2D(ss1.str(), ss2.str(), 80, -20., 20., 80, -20., 20.));
    ss1.str(std::string());
    ss1 << "position_siPixelClusters_mz_Disk_" << i;
    ss2.str(std::string());
    ss2 << "Clusters -Z Disk" << i << ";Global X (cm);Global Y (cm)";
    meClPosDiskmz.push_back(iBooker.book2D(ss1.str(), ss2.str(), 80, -20., 20., 80, -20., 20.));
  }

  // Book trend cluster plots for barrel and endcap. Lumisections for offline
  // and second for online - taken from strips
  iBooker.setCurrentFolder(topFolderName_ + "/Barrel");
  ss1.str(std::string());
  ss1 << "totalNumberOfClustersProfile_siPixelClusters_Barrel";
  ss2.str(std::string());
  ss2 << "Total number of barrel clusters profile;Lumisection;";
  meClusBarrelProf = iBooker.bookProfile(ss1.str(), ss2.str(), 2400, 0., 150, 0, 0, "");
  meClusBarrelProf->getTH1()->SetCanExtend(TH1::kAllAxes);

  iBooker.setCurrentFolder(topFolderName_ + "/Endcap");

  ss1.str(std::string());
  ss1 << "totalNumberOfClustersProfile_siPixelClusters_FPIX+";
  ss2.str(std::string());
  ss2 << "Total number of FPIX+ clusters profile;Lumisection;";
  meClusFpixPProf = iBooker.bookProfile(ss1.str(), ss2.str(), 2400, 0., 150, 0, 0, "");
  meClusFpixPProf->getTH1()->SetCanExtend(TH1::kAllAxes);

  ss1.str(std::string());
  ss1 << "totalNumberOfClustersProfile_siPixelClusters_FPIX-";
  ss2.str(std::string());
  ss2 << "Total number of FPIX- clusters profile;Lumisection;";
  meClusFpixMProf = iBooker.bookProfile(ss1.str(), ss2.str(), 2400, 0., 150, 0, 0, "");
  meClusFpixMProf->getTH1()->SetCanExtend(TH1::kAllAxes);

  iBooker.setCurrentFolder(topFolderName_ + "/Barrel");
  for (int i = 1; i <= noOfLayers; i++) {
    int ybins = -1;
    float ymin = 0.;
    float ymax = 0.;
    if (i == 1) {
      ybins = 42;
      ymin = -10.5;
      ymax = 10.5;
    }
    if (i == 2) {
      ybins = 66;
      ymin = -16.5;
      ymax = 16.5;
    }
    if (i == 3) {
      ybins = 90;
      ymin = -22.5;
      ymax = 22.5;
    }
    if (i == 4) {
      ybins = 130;
      ymin = -32.5;
      ymax = 32.5;
    }
    ss1.str(std::string());
    ss1 << "pix_bar Occ_roc_online_" + digisrc_.label() + "_layer_" << i;
    ss2.str(std::string());
    ss2 << "Pixel Barrel Occupancy, ROC level (Online): Layer " << i;
    meZeroRocBPIX.push_back(iBooker.book2D(ss1.str(), ss2.str(), 72, -4.5, 4.5, ybins, ymin, ymax));
    meZeroRocBPIX.at(i - 1)->setAxisTitle("ROC / Module", 1);
    meZeroRocBPIX.at(i - 1)->setAxisTitle("ROC / Ladder", 2);
  }

  iBooker.setCurrentFolder(topFolderName_ + "/Endcap");
  meZeroRocFPIX = iBooker.book2D(
      "ROC_endcap_occupancy", "Pixel Endcap Occupancy, ROC level (Online)", 72, -4.5, 4.5, 288, -12.5, 12.5);
  meZeroRocFPIX->setBinLabel(1, "Disk-2 Pnl2", 1);
  meZeroRocFPIX->setBinLabel(9, "Disk-2 Pnl1", 1);
  meZeroRocFPIX->setBinLabel(19, "Disk-1 Pnl2", 1);
  meZeroRocFPIX->setBinLabel(27, "Disk-1 Pnl1", 1);
  meZeroRocFPIX->setBinLabel(41, "Disk+1 Pnl1", 1);
  meZeroRocFPIX->setBinLabel(49, "Disk+1 Pnl2", 1);
  meZeroRocFPIX->setBinLabel(59, "Disk+2 Pnl1", 1);
  meZeroRocFPIX->setBinLabel(67, "Disk+2 Pnl2", 1);
  meZeroRocFPIX->setAxisTitle("Blades in Inner (>0) / Outer(<) Halves", 2);
  meZeroRocFPIX->setAxisTitle("ROC occupancy", 3);
}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelClusterSource::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  eventNo++;

  if (meClPosLayer.at(0) && meClPosLayer.at(0)->getEntries() > 150000) {
    for (int i = 0; i < noOfLayers; i++) {
      meClPosLayer.at(i)->Reset();
    }
    for (int i = 0; i < noOfDisks; i++) {
      meClPosDiskpz.at(i)->Reset();
      meClPosDiskmz.at(i)->Reset();
    }
  }

  // get input data
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> input;
  iEvent.getByToken(srcToken_, input);
  auto const &clustColl = *(input.product());

  edm::ESHandle<TrackerGeometry> pDD = iSetup.getHandle(trackerGeomToken_);
  const TrackerGeometry *tracker = &(*pDD);

  edm::Handle<edm::DetSetVector<PixelDigi>> digiinput;
  iEvent.getByToken(digisrcToken_, digiinput);
  const edm::DetSetVector<PixelDigi> diginp = *(digiinput.product());

  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(trackerTopoToken_);
  const TrackerTopology *pTT = tTopoHandle.product();

  int lumiSection = (int)iEvent.luminosityBlock();
  int nEventFpixClusters = 0;

  int nEventsBarrel = 0;
  int nEventsFPIXm = 0;
  int nEventsFPIXp = 0;

  std::map<uint32_t, SiPixelClusterModule *>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++) {
    int numberOfFpixClusters = (*struct_iter)
                                   .second->fill(*input,
                                                 pTT,
                                                 tracker,
                                                 &nEventsBarrel,
                                                 &nEventsFPIXp,
                                                 &nEventsFPIXm,
                                                 meClPosLayer,
                                                 meClPosDiskpz,
                                                 meClPosDiskmz,
                                                 modOn,
                                                 ladOn,
                                                 layOn,
                                                 phiOn,
                                                 bladeOn,
                                                 diskOn,
                                                 ringOn,
                                                 twoDimOn,
                                                 reducedSet,
                                                 smileyOn,
                                                 isUpgrade);
    nEventFpixClusters = nEventFpixClusters + numberOfFpixClusters;
  }

  if (nEventFpixClusters > bigEventSize) {
    if (bigFpixClusterEventRate) {
      bigFpixClusterEventRate->Fill(lumiSection, 1. / 23.);
    }
  }

  float trendVar = iEvent.orbitNumber() / 262144.0;  // lumisection : seconds - matches strip trend plot

  meClusBarrelProf->Fill(trendVar, nEventsBarrel);
  meClusFpixPProf->Fill(trendVar, nEventsFPIXp);
  meClusFpixMProf->Fill(trendVar, nEventsFPIXm);

  // std::cout<<"nEventFpixClusters: "<<nEventFpixClusters<<" , nLumiSecs:
  // "<<nLumiSecs<<" , nBigEvents: "<<nBigEvents<<std::endl;

  for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) {
    DetId detId = (*it)->geographicalId();

    // fill barrel
    if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
      getrococcupancy(detId, diginp, pTT, meZeroRocBPIX);
    }

    // fill endcap
    if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
      getrococcupancye(detId, clustColl, pTT, pDD, meZeroRocFPIX);
    }
  }

  // slow down...
  if (slowDown)
    usleep(10000);
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelClusterSource::buildStructure(const edm::EventSetup &iSetup) {
  LogInfo("PixelDQM") << " SiPixelClusterSource::buildStructure";
  edm::ESHandle<TrackerGeometry> pDD = iSetup.getHandle(trackerGeomTokenBeginRun_);

  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(trackerTopoTokenBeginRun_);
  const TrackerTopology *pTT = tTopoHandle.product();

  LogVerbatim("PixelDQM") << " *** Geometry node for TrackerGeom is  " << &(*pDD) << std::endl;
  LogVerbatim("PixelDQM") << " *** I have " << pDD->dets().size() << " detectors" << std::endl;
  LogVerbatim("PixelDQM") << " *** I have " << pDD->detTypes().size() << " types" << std::endl;

  for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) {
    if (dynamic_cast<PixelGeomDetUnit const *>((*it)) != nullptr) {
      DetId detId = (*it)->geographicalId();
      const GeomDetUnit *geoUnit = pDD->idToDetUnit(detId);
      const PixelGeomDetUnit *pixDet = dynamic_cast<const PixelGeomDetUnit *>(geoUnit);
      int nrows = (pixDet->specificTopology()).nrows();
      int ncols = (pixDet->specificTopology()).ncolumns();

      if ((detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) ||
          (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap))) {
        uint32_t id = detId();
        if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
          if (isPIB)
            continue;
          LogDebug("PixelDQM") << " ---> Adding Barrel Module " << detId.rawId() << endl;
          int layer = PixelBarrelName(DetId(id), pTT, isUpgrade).layerName();
          if (layer > noOfLayers)
            noOfLayers = layer;
          SiPixelClusterModule *theModule = new SiPixelClusterModule(id, ncols, nrows);
          thePixelStructure.insert(pair<uint32_t, SiPixelClusterModule *>(id, theModule));
        } else if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
          LogDebug("PixelDQM") << " ---> Adding Endcap Module " << detId.rawId() << endl;
          PixelEndcapName::HalfCylinder side = PixelEndcapName(DetId(id), pTT, isUpgrade).halfCylinder();
          int disk = PixelEndcapName(DetId(id), pTT, isUpgrade).diskName();
          if (disk > noOfDisks)
            noOfDisks = disk;
          int blade = PixelEndcapName(DetId(id), pTT, isUpgrade).bladeName();
          int panel = PixelEndcapName(DetId(id), pTT, isUpgrade).pannelName();
          int module = PixelEndcapName(DetId(id), pTT, isUpgrade).plaquetteName();
          char sside[80];
          sprintf(sside, "HalfCylinder_%i", side);
          char sdisk[80];
          sprintf(sdisk, "Disk_%i", disk);
          char sblade[80];
          sprintf(sblade, "Blade_%02i", blade);
          char spanel[80];
          sprintf(spanel, "Panel_%i", panel);
          char smodule[80];
          sprintf(smodule, "Module_%i", module);
          std::string side_str = sside;
          std::string disk_str = sdisk;
          bool mask = side_str.find("HalfCylinder_1") != string::npos ||
                      side_str.find("HalfCylinder_2") != string::npos ||
                      side_str.find("HalfCylinder_4") != string::npos || disk_str.find("Disk_2") != string::npos;
          // clutch to take all of FPIX, but no BPIX:
          mask = false;
          if (isPIB && mask)
            continue;
          SiPixelClusterModule *theModule = new SiPixelClusterModule(id, ncols, nrows);
          thePixelStructure.insert(pair<uint32_t, SiPixelClusterModule *>(id, theModule));
        }
      }
    }
  }
  LogInfo("PixelDQM") << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelClusterSource::bookMEs(DQMStore::IBooker &iBooker, const edm::EventSetup &iSetup) {
  // Get DQM interface
  iBooker.setCurrentFolder(topFolderName_);
  char title[256];
  snprintf(title,
           256,
           "Rate of events with >%i FPIX clusters;LumiSection;Rate of large "
           "FPIX events per LS [Hz]",
           bigEventSize);
  bigFpixClusterEventRate = iBooker.book1D("bigFpixClusterEventRate", title, 5000, 0., 5000.);

  std::map<uint32_t, SiPixelClusterModule *>::iterator struct_iter;

  SiPixelFolderOrganizer theSiPixelFolder(false);

  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(trackerTopoTokenBeginRun_);
  const TrackerTopology *pTT = tTopoHandle.product();

  for (struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++) {
    /// Create folder tree and book histograms
    if (modOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 0, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 0, twoDimOn, reducedSet, isUpgrade);
      } else {
        if (!isPIB)
          throw cms::Exception("LogicError") << "[SiPixelClusterSource::bookMEs] Creation of DQM folder "
                                                "failed";
      }
    }
    if (ladOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 1, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 1, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH LADDER-FOLDER\n";
      }
    }
    if (layOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 2, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 2, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH LAYER-FOLDER\n";
      }
    }
    if (phiOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 3, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 3, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH PHI-FOLDER\n";
      }
    }
    if (bladeOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 4, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 4, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH BLADE-FOLDER\n";
      }
    }
    if (diskOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 5, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 5, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH DISK-FOLDER\n";
      }
    }
    if (ringOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 6, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 6, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH RING-FOLDER\n";
      }
    }
    if (smileyOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 7, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 7, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH BARREL-FOLDER\n";
      }
    }
  }
}

void SiPixelClusterSource::getrococcupancy(DetId detId,
                                           const edm::DetSetVector<PixelDigi> &diginp,
                                           const TrackerTopology *const tTopo,
                                           std::vector<MonitorElement *> const &meinput) {
  edm::DetSetVector<PixelDigi>::const_iterator ipxsearch = diginp.find(detId);
  if (ipxsearch != diginp.end()) {
    // Look at digis now
    edm::DetSet<PixelDigi>::const_iterator pxdi;
    for (pxdi = ipxsearch->begin(); pxdi != ipxsearch->end(); pxdi++) {
      bool isHalfModule = PixelBarrelName(DetId(detId), tTopo, isUpgrade).isHalfModule();
      int DBlayer = PixelBarrelName(DetId(detId), tTopo, isUpgrade).layerName();
      int DBmodule = PixelBarrelName(DetId(detId), tTopo, isUpgrade).moduleName();
      int DBladder = PixelBarrelName(DetId(detId), tTopo, isUpgrade).ladderName();
      int DBshell = PixelBarrelName(DetId(detId), tTopo, isUpgrade).shell();

      // add sign to the modules
      if (DBshell == 1 || DBshell == 2) {
        DBmodule = -DBmodule;
      }
      if (DBshell == 1 || DBshell == 3) {
        DBladder = -DBladder;
      }

      int col = pxdi->column();
      int row = pxdi->row();

      float modsign = (float)DBmodule / (abs((float)DBmodule));
      float ladsign = (float)DBladder / (abs((float)DBladder));
      float rocx = ((float)col / (52. * 8.)) * modsign + ((float)DBmodule - (modsign)*0.5);
      float rocy = ((float)row / (80. * 2.)) * ladsign + ((float)DBladder - (ladsign)*0.5);

      // do the flip where need
      bool flip = false;
      if ((DBladder % 2 == 0) && (!isHalfModule)) {
        flip = true;
      }
      if ((flip) && (DBladder > 0)) {
        if ((((float)DBladder - (ladsign)*0.5) <= rocy) && (rocy < (float)DBladder)) {
          rocy = rocy + ladsign * 0.5;
        } else if ((((float)DBladder) <= rocy) && (rocy < ((float)DBladder + (ladsign)*0.5))) {
          rocy = rocy - ladsign * 0.5;
        }
      }

      // tweak border effect for negative modules/ladders
      if (modsign < 0) {
        rocx = rocx - 0.0001;
      }
      if (ladsign < 0) {
        rocy = rocy - 0.0001;
      } else {
        rocy = rocy + 0.0001;
      }
      if (abs(DBladder) == 1) {
        rocy = rocy + ladsign * 0.5;
      }  // take care of the half module
      meinput[DBlayer - 1]->Fill(rocx, rocy);
    }  // end of looping over pxdi
  }
}

void SiPixelClusterSource::getrococcupancye(DetId detId,
                                            const edmNew::DetSetVector<SiPixelCluster> &clustColl,
                                            const TrackerTopology *const pTT,
                                            edm::ESHandle<TrackerGeometry> pDD,
                                            MonitorElement *meinput) {
  edmNew::DetSetVector<SiPixelCluster>::const_iterator ipxsearch = clustColl.find(detId);
  if (ipxsearch != clustColl.end()) {
    // Look at clusters now
    edmNew::DetSet<SiPixelCluster>::const_iterator pxclust;
    for (pxclust = ipxsearch->begin(); pxclust != ipxsearch->end(); pxclust++) {
      const GeomDetUnit *geoUnit = pDD->idToDetUnit(detId);
      const PixelGeomDetUnit *pixDet = dynamic_cast<const PixelGeomDetUnit *>(geoUnit);
      const PixelTopology *topol = &(pixDet->specificTopology());
      LocalPoint clustlp = topol->localPosition(MeasurementPoint(pxclust->x(), pxclust->y()));
      GlobalPoint clustgp = geoUnit->surface().toGlobal(clustlp);

      float xclust = pxclust->x();
      float yclust = pxclust->y();
      float z = clustgp.z();

      int pxfside = PixelEndcapName(detId, pTT, isUpgrade).halfCylinder();
      int pxfpanel = PixelEndcapName(detId, pTT, isUpgrade).pannelName();
      int pxfmodule = PixelEndcapName(detId, pTT, isUpgrade).plaquetteName();
      int pxfdisk = PixelEndcapName(detId, pTT, isUpgrade).diskName();
      int pxfblade = PixelEndcapName(detId, pTT, isUpgrade).bladeName();

      if ((pxfside == 1) || (pxfside == 3)) {
        pxfblade = -1. * pxfblade;
      }

      if (z < 0.) {
        pxfdisk = -1. * pxfdisk;
      }

      int clu_sdpx = ((pxfdisk > 0) ? 1 : -1) * (2 * (abs(pxfdisk) - 1) + pxfpanel);
      int binselx = (pxfpanel == 1 && (pxfmodule == 1 || pxfmodule == 4))
                        ? (pxfmodule == 1)
                        : ((pxfpanel == 1 && xclust < 80.0) || (pxfpanel == 2 && xclust >= 80.0));
      int nperpan = 2 * pxfmodule + pxfpanel - 1 + binselx;
      int clu_roc_binx =
          ((pxfdisk > 0) ? nperpan : 9 - nperpan) + (clu_sdpx + 4) * 8 - 2 * ((abs(pxfdisk) == 1) ? pxfdisk : 0);

      int clu_roc_biny = -99.;
      int nrocly = pxfmodule + pxfpanel;
      for (int i = 0; i < nrocly; i++) {
        int j = (pxfdisk < 0) ? i : nrocly - 1 - i;
        if (yclust >= (j * 52.0) && yclust < ((j + 1) * 52.0))
          clu_roc_biny = 6 - nrocly + 2 * i + ((pxfblade > 0) ? pxfblade - 1 : pxfblade + 12) * 12 + 1;
      }
      if (pxfblade > 0) {
        clu_roc_biny = clu_roc_biny + 144;
      }

      meinput->setBinContent(clu_roc_binx, clu_roc_biny, meinput->getBinContent(clu_roc_binx, clu_roc_biny) + 1);
      meinput->setBinContent(
          clu_roc_binx, clu_roc_biny + 1, meinput->getBinContent(clu_roc_binx, clu_roc_biny + 1) + 1);
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(SiPixelClusterSource);
