// -*- C++ -*-
//
// Package:    SiPixelMonitorRecHits
// Class:      SiPixelRecHitSource
//
/**\class

 Description: Pixel DQM source for RecHits

 Implementation:
     Originally based on the code for Digis, adapted
        to read RecHits and create relevant histograms
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:
//
//
// Adapted by:  Keith Rose
//  	For use in SiPixelMonitorClient for RecHits
// Updated by: Lukas Wehrli
// for pixel offline DQM

#include "DQM/SiPixelMonitorRecHit/interface/SiPixelRecHitSource.h"
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
// DQM Framework
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQMServices/Core/interface/DQMStore.h"
// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// DataFormats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

//
#include <cstdlib>
#include <iostream>
#include <string>
using namespace std;
using namespace edm;

SiPixelRecHitSource::SiPixelRecHitSource(const edm::ParameterSet &iConfig)
    : conf_(iConfig),
      src_(consumes<SiPixelRecHitCollection>(conf_.getParameter<edm::InputTag>("src"))),
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
      isUpgrade(conf_.getUntrackedParameter<bool>("isUpgrade", false)) {
  firstRun = true;
  LogInfo("PixelDQM") << "SiPixelRecHitSource::SiPixelRecHitSource: Got DQM BackEnd interface" << endl;
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
}

SiPixelRecHitSource::~SiPixelRecHitSource() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  LogInfo("PixelDQM") << "SiPixelRecHitSource::~SiPixelRecHitSource: Destructor" << endl;
  std::map<uint32_t, SiPixelRecHitModule *>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++) {
    delete struct_iter->second;
    struct_iter->second = nullptr;
  }
}

void SiPixelRecHitSource::dqmBeginRun(const edm::Run &r, const edm::EventSetup &iSetup) {
  LogInfo("PixelDQM") << " SiPixelRecHitSource::beginJob - Initialisation ... " << std::endl;
  LogInfo("PixelDQM") << "Mod/Lad/Lay/Phi " << modOn << "/" << ladOn << "/" << layOn << "/" << phiOn << std::endl;
  LogInfo("PixelDQM") << "Blade/Disk/Ring" << bladeOn << "/" << diskOn << "/" << ringOn << std::endl;
  LogInfo("PixelDQM") << "2DIM IS " << twoDimOn << "\n";

  if (firstRun) {
    eventNo = 0;
    // Build map
    buildStructure(iSetup);
    // Book Monitoring Elements
    firstRun = false;
  }
}

void SiPixelRecHitSource::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, const edm::EventSetup &iSetup) {
  bookMEs(iBooker, iSetup);
}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelRecHitSource::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  eventNo++;
  // cout << eventNo << endl;
  // get input data
  edm::Handle<SiPixelRecHitCollection> recHitColl;
  iEvent.getByToken(src_, recHitColl);

  std::map<uint32_t, SiPixelRecHitModule *>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++) {
    uint32_t TheID = (*struct_iter).first;

    SiPixelRecHitCollection::const_iterator match = recHitColl->find(TheID);

    // if( pixelrechitRangeIteratorBegin == pixelrechitRangeIteratorEnd) {cout
    // << "oops" << endl;}
    float rechit_x = 0;
    float rechit_y = 0;
    int rechit_count = 0;

    if (match != recHitColl->end()) {
      SiPixelRecHitCollection::DetSet pixelrechitRange = *match;
      SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.begin();
      SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorEnd = pixelrechitRange.end();
      SiPixelRecHitCollection::DetSet::const_iterator pixeliter = pixelrechitRangeIteratorBegin;

      for (; pixeliter != pixelrechitRangeIteratorEnd; pixeliter++) {
        rechit_count++;
        // cout << TheID << endl;
        SiPixelRecHit::ClusterRef const &clust = pixeliter->cluster();
        int sizeX = (*clust).sizeX();
        // cout << sizeX << endl;
        int sizeY = (*clust).sizeY();
        // cout << sizeY << endl;
        LocalPoint lp = pixeliter->localPosition();
        rechit_x = lp.x();
        rechit_y = lp.y();

        LocalError lerr = pixeliter->localPositionError();
        float lerr_x = sqrt(lerr.xx());
        float lerr_y = sqrt(lerr.yy());
        // std::cout << "errors " << lerr_x << " " << lerr_y << std::endl;
        (*struct_iter)
            .second->fill(rechit_x,
                          rechit_y,
                          sizeX,
                          sizeY,
                          lerr_x,
                          lerr_y,
                          modOn,
                          ladOn,
                          layOn,
                          phiOn,
                          bladeOn,
                          diskOn,
                          ringOn,
                          twoDimOn,
                          reducedSet);
      }
    }
    if (rechit_count > 0)
      (*struct_iter).second->nfill(rechit_count, modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn);
  }

  // slow down...
  if (slowDown)
    usleep(10000);
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelRecHitSource::buildStructure(const edm::EventSetup &iSetup) {
  LogInfo("PixelDQM") << " SiPixelRecHitSource::buildStructure";

  edm::ESHandle<TrackerGeometry> pDD;
  edm::ESHandle<TrackerTopology> tTopoHandle;

  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);

  const TrackerTopology *pTT = tTopoHandle.product();

  LogVerbatim("PixelDQM") << " *** Geometry node for TrackerGeom is  " << &(*pDD) << std::endl;
  LogVerbatim("PixelDQM") << " *** I have " << pDD->dets().size() << " detectors" << std::endl;
  LogVerbatim("PixelDQM") << " *** I have " << pDD->detTypes().size() << " types" << std::endl;

  for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) {
    if (dynamic_cast<PixelGeomDetUnit const *>((*it)) != nullptr) {
      DetId detId = (*it)->geographicalId();

      if ((detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) ||
          (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap))) {
        uint32_t id = detId();

        if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
          if (isPIB)
            continue;
          LogDebug("PixelDQM") << " ---> Adding Barrel Module " << detId.rawId() << endl;
          SiPixelRecHitModule *theModule = new SiPixelRecHitModule(id);
          thePixelStructure.insert(pair<uint32_t, SiPixelRecHitModule *>(id, theModule));

        } else if ((detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap))) {
          LogDebug("PixelDQM") << " ---> Adding Endcap Module " << detId.rawId() << endl;

          PixelEndcapName::HalfCylinder side = PixelEndcapName(DetId(id), pTT, isUpgrade).halfCylinder();
          int disk = PixelEndcapName(DetId(id), pTT, isUpgrade).diskName();
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
          if (isPIB && mask)
            continue;

          SiPixelRecHitModule *theModule = new SiPixelRecHitModule(id);
          thePixelStructure.insert(pair<uint32_t, SiPixelRecHitModule *>(id, theModule));
        }
      }
    }
  }  // FOR_LOOP

  LogInfo("PixelDQM") << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelRecHitSource::bookMEs(DQMStore::IBooker &iBooker, const edm::EventSetup &iSetup) {
  std::map<uint32_t, SiPixelRecHitModule *>::iterator struct_iter;

  SiPixelFolderOrganizer theSiPixelFolder(false);

  for (struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++) {
    /// Create folder tree and book histograms
    if (modOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 0, isUpgrade)) {
        (*struct_iter).second->book(conf_, iBooker, iSetup, 0, twoDimOn, reducedSet, isUpgrade);
      } else {
        if (!isPIB)
          throw cms::Exception("LogicError") << "[SiPixelDigiSource::bookMEs] Creation of DQM folder failed";
      }
    }
    if (ladOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 1, isUpgrade)) {
        (*struct_iter).second->book(conf_, iBooker, iSetup, 1, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH LADDER-FOLDER\n";
      }
    }
    if (layOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 2, isUpgrade)) {
        (*struct_iter).second->book(conf_, iBooker, iSetup, 2, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH LAYER-FOLDER\n";
      }
    }
    if (phiOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 3, isUpgrade)) {
        (*struct_iter).second->book(conf_, iBooker, iSetup, 3, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH PHI-FOLDER\n";
      }
    }
    if (bladeOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 4, isUpgrade)) {
        (*struct_iter).second->book(conf_, iBooker, iSetup, 4, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH BLADE-FOLDER\n";
      }
    }
    if (diskOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 5, isUpgrade)) {
        (*struct_iter).second->book(conf_, iBooker, iSetup, 5, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH DISK-FOLDER\n";
      }
    }
    if (ringOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 6, isUpgrade)) {
        (*struct_iter).second->book(conf_, iBooker, iSetup, 6, twoDimOn, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH RING-FOLDER\n";
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(SiPixelRecHitSource);
