// -*- C++ -*-
//
// Package:    SiPixelMonitorDigi
// Class:      SiPixelDigiSource
//
/**\class 

 Description: Pixel DQM source for Digis

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:
//
//
#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiSource.h"
// Framework
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
// DQM Framework
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQMServices/Core/interface/DQMStore.h"
// Geometry
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
// DataFormats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
//
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;
using namespace edm;

SiPixelDigiSource::SiPixelDigiSource(const edm::ParameterSet& iConfig)
    : conf_(iConfig),
      src_(conf_.getParameter<edm::InputTag>("src")),
      saveFile(conf_.getUntrackedParameter<bool>("saveFile", false)),
      isPIB(conf_.getUntrackedParameter<bool>("isPIB", false)),
      slowDown(conf_.getUntrackedParameter<bool>("slowDown", false)),
      modOn(conf_.getUntrackedParameter<bool>("modOn", true)),
      perLSsaving(conf_.getUntrackedParameter<bool>("perLSsaving", false)),
      twoDimOn(conf_.getUntrackedParameter<bool>("twoDimOn", true)),
      twoDimModOn(conf_.getUntrackedParameter<bool>("twoDimModOn", true)),
      twoDimOnlyLayDisk(conf_.getUntrackedParameter<bool>("twoDimOnlyLayDisk", false)),
      hiRes(conf_.getUntrackedParameter<bool>("hiRes", false)),
      reducedSet(conf_.getUntrackedParameter<bool>("reducedSet", false)),
      ladOn(conf_.getUntrackedParameter<bool>("ladOn", false)),
      layOn(conf_.getUntrackedParameter<bool>("layOn", false)),
      phiOn(conf_.getUntrackedParameter<bool>("phiOn", false)),
      ringOn(conf_.getUntrackedParameter<bool>("ringOn", false)),
      bladeOn(conf_.getUntrackedParameter<bool>("bladeOn", false)),
      diskOn(conf_.getUntrackedParameter<bool>("diskOn", false)),
      bigEventSize(conf_.getUntrackedParameter<int>("bigEventSize", 1000)),
      isUpgrade(conf_.getUntrackedParameter<bool>("isUpgrade", false)),
      noOfLayers(0),
      noOfDisks(0) {
  //set Token(-s)
  srcToken_ = consumes<edm::DetSetVector<PixelDigi> >(conf_.getParameter<edm::InputTag>("src"));

  trackerTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  trackerTopoTokenBeginRun_ = esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>();
  trackerGeomTokenBeginRun_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>();

  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");

  firstRun = true;
  // find a FED# for the current detId:
  ifstream infile(edm::FileInPath("DQM/SiPixelMonitorClient/test/detId.dat").fullPath().c_str(), ios::in);
  int nModsInFile = 0;
  assert(!infile.fail());
  int nTOTmodules;
  if (isUpgrade) {
    nTOTmodules = 1856;
  } else {
    nTOTmodules = 1440;
  }
  while (!infile.eof() && nModsInFile < nTOTmodules) {
    infile >> I_name[nModsInFile] >> I_detId[nModsInFile] >> I_fedId[nModsInFile] >> I_linkId1[nModsInFile] >>
        I_linkId2[nModsInFile];
    nModsInFile++;
  }
  infile.close();

  LogInfo("PixelDQM") << "SiPixelDigiSource::SiPixelDigiSource: Got DQM BackEnd interface" << endl;
}

SiPixelDigiSource::~SiPixelDigiSource() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  LogInfo("PixelDQM") << "SiPixelDigiSource::~SiPixelDigiSource: Destructor" << endl;
}

std::shared_ptr<bool> SiPixelDigiSource::globalBeginLuminosityBlock(const edm::LuminosityBlock& lumi,
                                                                    const edm::EventSetup& iSetup) const {
  unsigned int currentLS = lumi.id().luminosityBlock();
  bool resetCounters = (currentLS % 10 == 0) ? true : false;
  return std::make_shared<bool>(resetCounters);
}

void SiPixelDigiSource::globalEndLuminosityBlock(const edm::LuminosityBlock& lb, edm::EventSetup const&) {
  int thisls = lb.id().luminosityBlock();
  const bool resetCounters = luminosityBlockCache(lb.index());

  float averageBPIXFed = float(nBPIXDigis) / 32.;
  float averageFPIXFed = float(nFPIXDigis) / 8.;

  if (averageDigiOccupancy) {
    for (int i = 0; i != 40; i++) {
      float averageOcc = 0.;
      if (i < 32) {
        if (averageBPIXFed > 0.)
          averageOcc = nDigisPerFed[i] / averageBPIXFed;
      } else {
        if (averageFPIXFed > 0.)
          averageOcc = nDigisPerFed[i] / averageFPIXFed;
      }
      if (!modOn) {
        averageDigiOccupancy->Fill(
            i,
            nDigisPerFed[i]);  //In offline we fill all digis and normalise at the end of the run for thread safe behaviour.
        avgfedDigiOccvsLumi->setBinContent(thisls, i + 1, nDigisPerFed[i]);  //Same plot vs lumi section
      }
      if (modOn) {
        if (thisls % 10 == 0)
          averageDigiOccupancy->Fill(
              i,
              averageOcc);  // "modOn" basically mean Online DQM, in this case fill histos with actual value of digi fraction per fed for each ten lumisections
        if (avgfedDigiOccvsLumi && thisls % 5 == 0)
          avgfedDigiOccvsLumi->setBinContent(
              int(thisls / 5),
              i + 1,
              averageOcc);  //fill with the mean over 5 lumisections, previous code was filling this histo only with last event of each 10th lumisection
      }
    }

    if (modOn && thisls % 10 == 0) {
      avgBarrelFedOccvsLumi->setBinContent(
          int(thisls / 10), averageBPIXFed);  //<NDigis> vs lumisection for barrel, filled every 10 lumi sections
      avgEndcapFedOccvsLumi->setBinContent(
          int(thisls / 10), averageFPIXFed);  //<NDigis> vs lumisection for endcap, filled every 10 lumi sections
    }
  }

  //reset counters

  if (modOn && resetCounters && averageDigiOccupancy) {
    nBPIXDigis = 0;
    nFPIXDigis = 0;
    for (int i = 0; i != 40; i++)
      nDigisPerFed[i] = 0;
  }

  if (!modOn && averageDigiOccupancy) {
    nBPIXDigis = 0;
    nFPIXDigis = 0;
    for (int i = 0; i != 40; i++)
      nDigisPerFed[i] = 0;
  }

  if (modOn && resetCounters) {
    ROCMapToReset = true;  //the ROC map is reset each 10 lumisections

    for (int i = 0; i < 2; i++)
      NzeroROCs[i] = 0;
    for (int i = 0; i < 2; i++)
      NloEffROCs[i] = 0;  //resetting also Zero and low eff. ROC counters

    NzeroROCs[1] = -672;
    NloEffROCs[1] =
        -672;  //this magic number derives by the way the endcap occupancy is filled, there are always 672 empty bins by construction

    //these bools are needed to count zero occupancy plots in the substructure only once each 10 LS
    DoZeroRocsBMO1 = true;
    DoZeroRocsBMO2 = true;
    DoZeroRocsBMO3 = true;

    DoZeroRocsBMI1 = true;
    DoZeroRocsBMI2 = true;
    DoZeroRocsBMI3 = true;

    DoZeroRocsBPO1 = true;
    DoZeroRocsBPO2 = true;
    DoZeroRocsBPO3 = true;

    DoZeroRocsBPI1 = true;
    DoZeroRocsBPI2 = true;
    DoZeroRocsBPI3 = true;

    DoZeroRocsFPO1 = true;
    DoZeroRocsFPO2 = true;

    DoZeroRocsFMO1 = true;
    DoZeroRocsFMO2 = true;

    DoZeroRocsFPI1 = true;
    DoZeroRocsFPI2 = true;

    DoZeroRocsFMI1 = true;
    DoZeroRocsFMI2 = true;
  }
}

void SiPixelDigiSource::dqmBeginRun(const edm::Run& r, const edm::EventSetup& iSetup) {
  LogInfo("PixelDQM") << " SiPixelDigiSource::beginJob - Initialisation ... " << std::endl;
  LogInfo("PixelDQM") << "Mod/Lad/Lay/Phi " << modOn << "/" << ladOn << "/" << layOn << "/" << phiOn << std::endl;
  LogInfo("PixelDQM") << "Blade/Disk/Ring" << bladeOn << "/" << diskOn << "/" << ringOn << std::endl;
  LogInfo("PixelDQM") << "2DIM IS " << twoDimOn << " and set to high resolution? " << hiRes << "\n";

  if (firstRun) {
    nBigEvents = 0;
    nBPIXDigis = 0;
    nFPIXDigis = 0;
    for (int i = 0; i != 40; i++)
      nDigisPerFed[i] = 0;
    for (int i = 0; i != 4; i++)
      nDigisPerDisk[i] = 0;
    nDP1P1M1 = 0;
    nDP1P1M2 = 0;
    nDP1P1M3 = 0;
    nDP1P1M4 = 0;
    nDP1P2M1 = 0;
    nDP1P2M2 = 0;
    nDP1P2M3 = 0;
    nDP2P1M1 = 0;
    nDP2P1M2 = 0;
    nDP2P1M3 = 0;
    nDP2P1M4 = 0;
    nDP2P2M1 = 0;
    nDP2P2M2 = 0;
    nDP2P2M3 = 0;
    nDP3P1M1 = 0;
    nDP3P2M1 = 0;
    nDM1P1M1 = 0;
    nDM1P1M2 = 0;
    nDM1P1M3 = 0;
    nDM1P1M4 = 0;
    nDM1P2M1 = 0;
    nDM1P2M2 = 0;
    nDM1P2M3 = 0;
    nDM2P1M1 = 0;
    nDM2P1M2 = 0;
    nDM2P1M3 = 0;
    nDM2P1M4 = 0;
    nDM2P2M1 = 0;
    nDM2P2M2 = 0;
    nDM2P2M3 = 0;
    nDM3P1M1 = 0;
    nDM3P2M1 = 0;
    nL1M1 = 0;
    nL1M2 = 0;
    nL1M3 = 0;
    nL1M4 = 0;
    nL2M1 = 0;
    nL2M2 = 0;
    nL2M3 = 0;
    nL2M4 = 0;
    nL3M1 = 0;
    nL3M2 = 0;
    nL3M3 = 0;
    nL3M4 = 0;
    nL4M1 = 0;
    nL4M2 = 0;
    nL4M3 = 0;
    nL4M4 = 0;

    ROCMapToReset = false;

    DoZeroRocsBMO1 = false;
    DoZeroRocsBMO2 = false;
    DoZeroRocsBMO3 = false;

    DoZeroRocsBMI1 = false;
    DoZeroRocsBMI2 = false;
    DoZeroRocsBMI3 = false;

    DoZeroRocsBPO1 = false;
    DoZeroRocsBPO2 = false;
    DoZeroRocsBPO3 = false;

    DoZeroRocsBPI1 = false;
    DoZeroRocsBPI2 = false;
    DoZeroRocsBPI3 = false;

    DoZeroRocsFPO1 = false;
    DoZeroRocsFPO2 = false;

    DoZeroRocsFMO1 = false;
    DoZeroRocsFMO2 = false;

    DoZeroRocsFPI1 = false;
    DoZeroRocsFPI2 = false;

    DoZeroRocsFMI1 = false;
    DoZeroRocsFMI2 = false;

    for (int i = 0; i < 2; i++)
      NzeroROCs[i] = 0;
    for (int i = 0; i < 2; i++)
      NloEffROCs[i] = 0;

    // Build map
    buildStructure(iSetup);
    // Book Monitoring Elements
    firstRun = false;
  }
}

void SiPixelDigiSource::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, const edm::EventSetup& iSetup) {
  bookMEs(iBooker, iSetup);
}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelDigiSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(trackerTopoToken_);
  const TrackerTopology* pTT = tTopoHandle.product();

  // get input data
  edm::Handle<edm::DetSetVector<PixelDigi> > input;
  iEvent.getByToken(srcToken_, input);
  if (!input.isValid())
    return;

  int bx = iEvent.bunchCrossing();

  int lumiSection = (int)iEvent.luminosityBlock();
  int nEventDigis = 0;
  int nActiveModules = 0;

  std::map<uint32_t, SiPixelDigiModule*>::iterator struct_iter;
  for (int i = 0; i != 192; i++)
    numberOfDigis[i] = 0;
  for (int i = 0; i != 1152; i++)
    nDigisPerChan[i] = 0;
  for (int i = 0; i != 4; i++)
    nDigisPerDisk[i] = 0;

  for (struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++) {
    int numberOfDigisMod = (*struct_iter)
                               .second->fill(*input,
                                             pTT,
                                             meNDigisCOMBBarrel_,
                                             meNDigisCHANBarrel_,
                                             meNDigisCHANBarrelLs_,
                                             meNDigisCOMBEndcap_,
                                             modOn,
                                             ladOn,
                                             layOn,
                                             phiOn,
                                             bladeOn,
                                             diskOn,
                                             ringOn,
                                             twoDimOn,
                                             reducedSet,
                                             twoDimModOn,
                                             twoDimOnlyLayDisk,
                                             nDigisA,
                                             nDigisB,
                                             isUpgrade);

    bool barrel = DetId((*struct_iter).first).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
    bool endcap = DetId((*struct_iter).first).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);

    if (numberOfDigisMod > 0) {
      nEventDigis = nEventDigis + numberOfDigisMod;
      nActiveModules++;
      int nBPiXmodules;
      int nTOTmodules;
      if (isUpgrade) {
        nBPiXmodules = 1184;
        nTOTmodules = 1856;
      } else {
        nBPiXmodules = 768;
        nTOTmodules = 1440;
      }
      if (barrel) {  // Barrel
        int layer = PixelBarrelName(DetId((*struct_iter).first), pTT, isUpgrade).layerName();
        PixelBarrelName::Shell shell = PixelBarrelName(DetId((*struct_iter).first), pTT, isUpgrade).shell();

        //Count Zero Occ Rocs in Barrel in the first event after each 10 Ls
        if (ROCMapToReset && shell == PixelBarrelName::mO && twoDimOnlyLayDisk) {
          if (DoZeroRocsBMO1 && layer == 1)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBMO1, (*struct_iter).second);
          if (DoZeroRocsBMO2 && layer == 2)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBMO2, (*struct_iter).second);
          if (DoZeroRocsBMO3 && layer == 3)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBMO3, (*struct_iter).second);
        } else if (ROCMapToReset && shell == PixelBarrelName::mI && twoDimOnlyLayDisk) {
          if (DoZeroRocsBMI1 && layer == 1)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBMI1, (*struct_iter).second);
          if (DoZeroRocsBMI2 && layer == 2)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBMI2, (*struct_iter).second);
          if (DoZeroRocsBMI3 && layer == 3)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBMI3, (*struct_iter).second);
        } else if (ROCMapToReset && shell == PixelBarrelName::pO && twoDimOnlyLayDisk) {
          if (DoZeroRocsBPO1 && layer == 1)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBPO1, (*struct_iter).second);
          if (DoZeroRocsBPO2 && layer == 2)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBPO2, (*struct_iter).second);
          if (DoZeroRocsBPO3 && layer == 3)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBPO3, (*struct_iter).second);
        } else if (ROCMapToReset && shell == PixelBarrelName::pI && twoDimOnlyLayDisk) {
          if (DoZeroRocsBPI1 && layer == 1)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBPI1, (*struct_iter).second);
          if (DoZeroRocsBPI2 && layer == 2)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBPI2, (*struct_iter).second);
          if (DoZeroRocsBPI3 && layer == 3)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsBPI3, (*struct_iter).second);
        }

        nBPIXDigis = nBPIXDigis + numberOfDigisMod;
        for (int i = 0; i != nBPiXmodules; ++i) {
          if ((*struct_iter).first == I_detId[i]) {
            nDigisPerFed[I_fedId[i]] = nDigisPerFed[I_fedId[i]] + numberOfDigisMod;
            int index1 = 0;
            int index2 = 0;
            if (I_linkId1[i] > 0)
              index1 = I_fedId[i] * 36 + (I_linkId1[i] - 1);
            if (I_linkId2[i] > 0)
              index2 = I_fedId[i] * 36 + (I_linkId2[i] - 1);
            if (nDigisA > 0 && I_linkId1[i] > 0)
              nDigisPerChan[index1] = nDigisPerChan[index1] + nDigisA;
            if (nDigisB > 0 && I_linkId2[i] > 0)
              nDigisPerChan[index2] = nDigisPerChan[index2] + nDigisB;
            i = (nBPiXmodules - 1);
          }
        }
      } else if (endcap && !isUpgrade) {  // Endcap
        nFPIXDigis = nFPIXDigis + numberOfDigisMod;
        PixelEndcapName::HalfCylinder side =
            PixelEndcapName(DetId((*struct_iter).first), pTT, isUpgrade).halfCylinder();
        int disk = PixelEndcapName(DetId((*struct_iter).first), pTT, isUpgrade).diskName();
        int blade = PixelEndcapName(DetId((*struct_iter).first), pTT, isUpgrade).bladeName();
        int panel = PixelEndcapName(DetId((*struct_iter).first), pTT, isUpgrade).pannelName();
        int module = PixelEndcapName(DetId((*struct_iter).first), pTT, isUpgrade).plaquetteName();
        int iter = 0;
        int i = 0;

        //count Zero Occupancy ROCs in Endcap in the first event after each 10 Ls
        if (ROCMapToReset && side == PixelEndcapName::mO && twoDimOnlyLayDisk) {
          if (DoZeroRocsFMO1 && disk == 1)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsFMO1, (*struct_iter).second);
          if (DoZeroRocsFMO2 && disk == 2)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsFMO2, (*struct_iter).second);
        } else if (ROCMapToReset && side == PixelEndcapName::mI && twoDimOnlyLayDisk) {
          if (DoZeroRocsFMI1 && disk == 1)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsFMI1, (*struct_iter).second);
          if (DoZeroRocsFMI2 && disk == 2)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsFMI2, (*struct_iter).second);
        } else if (ROCMapToReset && side == PixelEndcapName::pO && twoDimOnlyLayDisk) {
          if (DoZeroRocsFPO1 && disk == 1)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsFPO1, (*struct_iter).second);
          if (DoZeroRocsFPO2 && disk == 2)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsFPO2, (*struct_iter).second);
        } else if (ROCMapToReset && side == PixelEndcapName::pI && twoDimOnlyLayDisk) {
          if (DoZeroRocsFPI1 && disk == 1)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsFPI1, (*struct_iter).second);
          if (DoZeroRocsFPI2 && disk == 2)
            CountZeroROCsInSubstructure(barrel, DoZeroRocsFPI2, (*struct_iter).second);
        }

        if (side == PixelEndcapName::mI) {
          if (disk == 1) {
            i = 0;
            if (panel == 1) {
              if (module == 1)
                nDM1P1M1 += numberOfDigisMod;
              else if (module == 2)
                nDM1P1M2 += numberOfDigisMod;
              else if (module == 3)
                nDM1P1M3 += numberOfDigisMod;
              else if (module == 4)
                nDM1P1M4 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDM1P2M1 += numberOfDigisMod;
              else if (module == 2)
                nDM1P2M2 += numberOfDigisMod;
              else if (module == 3)
                nDM1P2M3 += numberOfDigisMod;
            }
            if (blade < 13 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 2) {
            i = 24;
            if (panel == 1) {
              if (module == 1)
                nDM2P1M1 += numberOfDigisMod;
              else if (module == 2)
                nDM2P1M2 += numberOfDigisMod;
              else if (module == 3)
                nDM2P1M3 += numberOfDigisMod;
              else if (module == 4)
                nDM2P1M4 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDM2P2M1 += numberOfDigisMod;
              else if (module == 2)
                nDM2P2M2 += numberOfDigisMod;
              else if (module == 3)
                nDM2P2M3 += numberOfDigisMod;
            }
            if (blade < 13 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          }
        } else if (side == PixelEndcapName::mO) {
          if (disk == 1) {
            i = 48;
            if (panel == 1) {
              if (module == 1)
                nDM1P1M1 += numberOfDigisMod;
              else if (module == 2)
                nDM1P1M2 += numberOfDigisMod;
              else if (module == 3)
                nDM1P1M3 += numberOfDigisMod;
              else if (module == 4)
                nDM1P1M4 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDM1P2M1 += numberOfDigisMod;
              else if (module == 2)
                nDM1P2M2 += numberOfDigisMod;
              else if (module == 3)
                nDM1P2M3 += numberOfDigisMod;
            }
            if (blade < 13 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 2) {
            i = 72;
            if (panel == 1) {
              if (module == 1)
                nDM2P1M1 += numberOfDigisMod;
              else if (module == 2)
                nDM2P1M2 += numberOfDigisMod;
              else if (module == 3)
                nDM2P1M3 += numberOfDigisMod;
              else if (module == 4)
                nDM2P1M4 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDM2P2M1 += numberOfDigisMod;
              else if (module == 2)
                nDM2P2M2 += numberOfDigisMod;
              else if (module == 3)
                nDM2P2M3 += numberOfDigisMod;
            }
            if (blade < 13 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          }
        } else if (side == PixelEndcapName::pI) {
          if (disk == 1) {
            i = 96;
            if (panel == 1) {
              if (module == 1)
                nDP1P1M1 += numberOfDigisMod;
              else if (module == 2)
                nDP1P1M2 += numberOfDigisMod;
              else if (module == 3)
                nDP1P1M3 += numberOfDigisMod;
              else if (module == 4)
                nDP1P1M4 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDP1P2M1 += numberOfDigisMod;
              else if (module == 2)
                nDP1P2M2 += numberOfDigisMod;
              else if (module == 3)
                nDP1P2M3 += numberOfDigisMod;
            }
            if (blade < 13 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 2) {
            i = 120;
            if (panel == 1) {
              if (module == 1)
                nDP2P1M1 += numberOfDigisMod;
              else if (module == 2)
                nDP2P1M2 += numberOfDigisMod;
              else if (module == 3)
                nDP2P1M3 += numberOfDigisMod;
              else if (module == 4)
                nDP2P1M4 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDP2P2M1 += numberOfDigisMod;
              else if (module == 2)
                nDP2P2M2 += numberOfDigisMod;
              else if (module == 3)
                nDP2P2M3 += numberOfDigisMod;
            }
            if (blade < 13 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          }
        } else if (side == PixelEndcapName::pO) {
          if (disk == 1) {
            i = 144;
            if (panel == 1) {
              if (module == 1)
                nDP1P1M1 += numberOfDigisMod;
              else if (module == 2)
                nDP1P1M2 += numberOfDigisMod;
              else if (module == 3)
                nDP1P1M3 += numberOfDigisMod;
              else if (module == 4)
                nDP1P1M4 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDP1P2M1 += numberOfDigisMod;
              else if (module == 2)
                nDP1P2M2 += numberOfDigisMod;
              else if (module == 3)
                nDP1P2M3 += numberOfDigisMod;
            }
            if (blade < 13 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 2) {
            i = 168;
            if (panel == 1) {
              if (module == 1)
                nDP2P1M1 += numberOfDigisMod;
              else if (module == 2)
                nDP2P1M2 += numberOfDigisMod;
              else if (module == 3)
                nDP2P1M3 += numberOfDigisMod;
              else if (module == 4)
                nDP2P1M4 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDP2P2M1 += numberOfDigisMod;
              else if (module == 2)
                nDP2P2M2 += numberOfDigisMod;
              else if (module == 3)
                nDP2P2M3 += numberOfDigisMod;
            }
            if (blade < 13 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          }
        }
        numberOfDigis[iter] = numberOfDigis[iter] + numberOfDigisMod;

        for (int i = nBPiXmodules; i != nTOTmodules; i++) {
          if ((*struct_iter).first == I_detId[i]) {
            nDigisPerFed[I_fedId[i]] = nDigisPerFed[I_fedId[i]] + numberOfDigisMod;
            i = nTOTmodules - 1;
          }
        }
      }  //endif Barrel/(Endcap && !isUpgrade)
      else if (endcap && isUpgrade) {
        nFPIXDigis = nFPIXDigis + numberOfDigisMod;
        PixelEndcapName::HalfCylinder side =
            PixelEndcapName(DetId((*struct_iter).first), pTT, isUpgrade).halfCylinder();
        int disk = PixelEndcapName(DetId((*struct_iter).first), pTT, isUpgrade).diskName();
        int blade = PixelEndcapName(DetId((*struct_iter).first), pTT, isUpgrade).bladeName();
        int panel = PixelEndcapName(DetId((*struct_iter).first), pTT, isUpgrade).pannelName();
        int module = PixelEndcapName(DetId((*struct_iter).first), pTT, isUpgrade).plaquetteName();

        int iter = 0;
        int i = 0;
        if (side == PixelEndcapName::mI) {
          if (disk == 1) {
            i = 0;
            if (panel == 1) {
              if (module == 1)
                nDM1P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDM1P2M1 += numberOfDigisMod;
            }
            if (blade < 12 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 2) {
            i = 22;
            if (panel == 1) {
              if (module == 1)
                nDM2P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDM2P2M1 += numberOfDigisMod;
            }
            if (blade < 12 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 3) {
            i = 44;
            if (panel == 1) {
              if (module == 1)
                nDM3P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDM3P2M1 += numberOfDigisMod;
            }
            if (blade < 12 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          }
        } else if (side == PixelEndcapName::mO) {
          if (disk == 1) {
            i = 66;
            if (panel == 1) {
              if (module == 1)
                nDM1P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDM1P2M1 += numberOfDigisMod;
            }
            if (blade < 18 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 2) {
            i = 100;
            if (panel == 1) {
              if (module == 1)
                nDM2P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDM2P2M1 += numberOfDigisMod;
            }
            if (blade < 18 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 3) {
            i = 134;
            if (panel == 1) {
              if (module == 1)
                nDM3P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDM3P2M1 += numberOfDigisMod;
            }
            if (blade < 18 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          }
        } else if (side == PixelEndcapName::pI) {
          if (disk == 1) {
            i = 168;
            if (panel == 1) {
              if (module == 1)
                nDP1P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDP1P2M1 += numberOfDigisMod;
            }
            if (blade < 12 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 2) {
            i = 190;
            if (panel == 1) {
              if (module == 1)
                nDP2P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDP2P2M1 += numberOfDigisMod;
            }
            if (blade < 12 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 3) {
            i = 212;
            if (panel == 1) {
              if (module == 1)
                nDP3P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDP3P2M1 += numberOfDigisMod;
            }
            if (blade < 12 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          }
        } else if (side == PixelEndcapName::pO) {
          if (disk == 1) {
            i = 234;
            if (panel == 1) {
              if (module == 1)
                nDP1P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDP1P2M1 += numberOfDigisMod;
            }
            if (blade < 18 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 2) {
            i = 268;
            if (panel == 1) {
              if (module == 1)
                nDP2P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDP2P2M1 += numberOfDigisMod;
            }
            if (blade < 18 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          } else if (disk == 3) {
            i = 302;
            if (panel == 1) {
              if (module == 1)
                nDP3P1M1 += numberOfDigisMod;
            } else if (panel == 2) {
              if (module == 1)
                nDP3P2M1 += numberOfDigisMod;
            }
            if (blade < 18 && blade > 0 && (panel == 1 || panel == 2))
              iter = i + 2 * (blade - 1) + (panel - 1);
          }
        }
        numberOfDigis[iter] = numberOfDigis[iter] + numberOfDigisMod;
        for (int i = nBPiXmodules; i != nTOTmodules; i++) {
          if ((*struct_iter).first == I_detId[i]) {
            nDigisPerFed[I_fedId[i]] = nDigisPerFed[I_fedId[i]] + numberOfDigisMod;
            i = nTOTmodules - 1;
          }
        }
      }  //endif(Endcap && isUpgrade)
    }    // endif any digis in this module
  }      // endfor loop over all modules

  if (lumiSection % 10 == 0 && ROCMapToReset) {
    for (int i = 0; i < 2; ++i)
      NloEffROCs[i] = NloEffROCs[i] - NzeroROCs[i];
    if (noOccROCsBarrel)
      noOccROCsBarrel->setBinContent(lumiSection / 10, NzeroROCs[0]);
    if (loOccROCsBarrel)
      loOccROCsBarrel->setBinContent(lumiSection / 10, NloEffROCs[0]);
    if (noOccROCsEndcap)
      noOccROCsEndcap->setBinContent(lumiSection / 10, NzeroROCs[1]);
    if (loOccROCsEndcap)
      loOccROCsEndcap->setBinContent(lumiSection / 10, NloEffROCs[1]);
    ROCMapToReset =
        false;  // in this way the ROC maps are reset for one event only (the first event in LS multiple of 10
  }

  if (noOfDisks == 2) {  // if (!isUpgrade)
    if (meNDigisCHANEndcap_) {
      for (int j = 0; j != 192; j++)
        if (numberOfDigis[j] > 0)
          meNDigisCHANEndcap_->Fill((float)numberOfDigis[j]);
    }
    if (meNDigisCHANEndcapDms_.at(0)) {
      for (int j = 0; j != 72; j++)
        if ((j < 24 || j > 47) && numberOfDigis[j] > 0)
          meNDigisCHANEndcapDms_.at(0)->Fill((float)numberOfDigis[j]);
    }
    if (meNDigisCHANEndcapDms_.at(1)) {
      for (int j = 24; j != 96; j++)
        if ((j < 48 || j > 71) && numberOfDigis[j] > 0)
          meNDigisCHANEndcapDms_.at(1)->Fill((float)numberOfDigis[j]);
    }
    if (meNDigisCHANEndcapDps_.at(0)) {
      for (int j = 96; j != 168; j++)
        if ((j < 120 || j > 143) && numberOfDigis[j] > 0)
          meNDigisCHANEndcapDps_.at(0)->Fill((float)numberOfDigis[j]);
    }
    if (meNDigisCHANEndcapDps_.at(1)) {
      for (int j = 120; j != 192; j++)
        if ((j < 144 || j > 167) && numberOfDigis[j] > 0)
          meNDigisCHANEndcapDps_.at(1)->Fill((float)numberOfDigis[j]);
    }
  } else if (noOfDisks == 3) {  // else if (isUpgrade)
    if (meNDigisCHANEndcap_) {
      for (int j = 0; j != 336; j++)
        if (numberOfDigis[j] > 0)
          meNDigisCHANEndcap_->Fill((float)numberOfDigis[j]);
    }
    if (meNDigisCHANEndcapDms_.at(0)) {
      for (int j = 0; j != 100; j++)
        if ((j < 22 || j > 65) && numberOfDigis[j] > 0)
          meNDigisCHANEndcapDms_.at(0)->Fill((float)numberOfDigis[j]);
    }
    if (meNDigisCHANEndcapDms_.at(1)) {
      for (int j = 22; j != 134; j++)
        if ((j < 44 || j > 99) && numberOfDigis[j] > 0)
          meNDigisCHANEndcapDms_.at(1)->Fill((float)numberOfDigis[j]);
    }
    if (meNDigisCHANEndcapDms_.at(2)) {
      for (int j = 44; j != 168; j++)
        if ((j < 66 || j > 133) && numberOfDigis[j] > 0)
          meNDigisCHANEndcapDms_.at(2)->Fill((float)numberOfDigis[j]);
    }
    if (meNDigisCHANEndcapDps_.at(0)) {
      for (int j = 168; j != 268; j++)
        if ((j < 190 || j > 233) && numberOfDigis[j] > 0)
          meNDigisCHANEndcapDps_.at(0)->Fill((float)numberOfDigis[j]);
    }
    if (meNDigisCHANEndcapDps_.at(1)) {
      for (int j = 190; j != 302; j++)
        if ((j < 212 || j > 267) && numberOfDigis[j] > 0)
          meNDigisCHANEndcapDps_.at(1)->Fill((float)numberOfDigis[j]);
    }
    if (meNDigisCHANEndcapDps_.at(2)) {
      for (int j = 212; j != 336; j++)
        if ((j < 234 || j > 301) && numberOfDigis[j] > 0)
          meNDigisCHANEndcapDps_.at(2)->Fill((float)numberOfDigis[j]);
    }
  }

  if (meNDigisCHANBarrelCh1_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 0] > 0)
        meNDigisCHANBarrelCh1_->Fill((float)nDigisPerChan[i * 36 + 0]);
  }
  if (meNDigisCHANBarrelCh2_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 1] > 0)
        meNDigisCHANBarrelCh2_->Fill((float)nDigisPerChan[i * 36 + 1]);
  }
  if (meNDigisCHANBarrelCh3_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 2] > 0)
        meNDigisCHANBarrelCh3_->Fill((float)nDigisPerChan[i * 36 + 2]);
  }
  if (meNDigisCHANBarrelCh4_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 3] > 0)
        meNDigisCHANBarrelCh4_->Fill((float)nDigisPerChan[i * 36 + 3]);
  }
  if (meNDigisCHANBarrelCh5_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 4] > 0)
        meNDigisCHANBarrelCh5_->Fill((float)nDigisPerChan[i * 36 + 4]);
  }
  if (meNDigisCHANBarrelCh6_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 5] > 0)
        meNDigisCHANBarrelCh6_->Fill((float)nDigisPerChan[i * 36 + 5]);
  }
  if (meNDigisCHANBarrelCh7_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 6] > 0)
        meNDigisCHANBarrelCh7_->Fill((float)nDigisPerChan[i * 36 + 6]);
  }
  if (meNDigisCHANBarrelCh8_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 7] > 0)
        meNDigisCHANBarrelCh8_->Fill((float)nDigisPerChan[i * 36 + 7]);
  }
  if (meNDigisCHANBarrelCh9_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 8] > 0)
        meNDigisCHANBarrelCh9_->Fill((float)nDigisPerChan[i * 36 + 8]);
  }
  if (meNDigisCHANBarrelCh10_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 9] > 0)
        meNDigisCHANBarrelCh10_->Fill((float)nDigisPerChan[i * 36 + 9]);
  }
  if (meNDigisCHANBarrelCh11_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 10] > 0)
        meNDigisCHANBarrelCh11_->Fill((float)nDigisPerChan[i * 36 + 10]);
  }
  if (meNDigisCHANBarrelCh12_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 11] > 0)
        meNDigisCHANBarrelCh12_->Fill((float)nDigisPerChan[i * 36 + 11]);
  }
  if (meNDigisCHANBarrelCh13_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 12] > 0)
        meNDigisCHANBarrelCh13_->Fill((float)nDigisPerChan[i * 36 + 12]);
  }
  if (meNDigisCHANBarrelCh14_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 13] > 0)
        meNDigisCHANBarrelCh14_->Fill((float)nDigisPerChan[i * 36 + 13]);
  }
  if (meNDigisCHANBarrelCh15_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 14] > 0)
        meNDigisCHANBarrelCh15_->Fill((float)nDigisPerChan[i * 36 + 14]);
  }
  if (meNDigisCHANBarrelCh16_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 15] > 0)
        meNDigisCHANBarrelCh16_->Fill((float)nDigisPerChan[i * 36 + 15]);
  }
  if (meNDigisCHANBarrelCh17_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 16] > 0)
        meNDigisCHANBarrelCh17_->Fill((float)nDigisPerChan[i * 36 + 16]);
  }
  if (meNDigisCHANBarrelCh18_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 17] > 0)
        meNDigisCHANBarrelCh18_->Fill((float)nDigisPerChan[i * 36 + 17]);
  }
  if (meNDigisCHANBarrelCh19_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 18] > 0)
        meNDigisCHANBarrelCh19_->Fill((float)nDigisPerChan[i * 36 + 18]);
  }
  if (meNDigisCHANBarrelCh20_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 19] > 0)
        meNDigisCHANBarrelCh20_->Fill((float)nDigisPerChan[i * 36 + 19]);
  }
  if (meNDigisCHANBarrelCh21_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 20] > 0)
        meNDigisCHANBarrelCh21_->Fill((float)nDigisPerChan[i * 36 + 20]);
  }
  if (meNDigisCHANBarrelCh22_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 21] > 0)
        meNDigisCHANBarrelCh22_->Fill((float)nDigisPerChan[i * 36 + 21]);
  }
  if (meNDigisCHANBarrelCh23_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 22] > 0)
        meNDigisCHANBarrelCh23_->Fill((float)nDigisPerChan[i * 36 + 22]);
  }
  if (meNDigisCHANBarrelCh24_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 23] > 0)
        meNDigisCHANBarrelCh24_->Fill((float)nDigisPerChan[i * 36 + 23]);
  }
  if (meNDigisCHANBarrelCh25_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 24] > 0)
        meNDigisCHANBarrelCh25_->Fill((float)nDigisPerChan[i * 36 + 24]);
  }
  if (meNDigisCHANBarrelCh26_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 25] > 0)
        meNDigisCHANBarrelCh26_->Fill((float)nDigisPerChan[i * 36 + 25]);
  }
  if (meNDigisCHANBarrelCh27_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 26] > 0)
        meNDigisCHANBarrelCh27_->Fill((float)nDigisPerChan[i * 36 + 26]);
  }
  if (meNDigisCHANBarrelCh28_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 27] > 0)
        meNDigisCHANBarrelCh28_->Fill((float)nDigisPerChan[i * 36 + 27]);
  }
  if (meNDigisCHANBarrelCh29_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 28] > 0)
        meNDigisCHANBarrelCh29_->Fill((float)nDigisPerChan[i * 36 + 28]);
  }
  if (meNDigisCHANBarrelCh30_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 29] > 0)
        meNDigisCHANBarrelCh30_->Fill((float)nDigisPerChan[i * 36 + 29]);
  }
  if (meNDigisCHANBarrelCh31_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 30] > 0)
        meNDigisCHANBarrelCh31_->Fill((float)nDigisPerChan[i * 36 + 30]);
  }
  if (meNDigisCHANBarrelCh32_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 31] > 0)
        meNDigisCHANBarrelCh32_->Fill((float)nDigisPerChan[i * 36 + 31]);
  }
  if (meNDigisCHANBarrelCh33_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 32] > 0)
        meNDigisCHANBarrelCh33_->Fill((float)nDigisPerChan[i * 36 + 32]);
  }
  if (meNDigisCHANBarrelCh34_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 33] > 0)
        meNDigisCHANBarrelCh34_->Fill((float)nDigisPerChan[i * 36 + 33]);
  }
  if (meNDigisCHANBarrelCh35_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 34] > 0)
        meNDigisCHANBarrelCh35_->Fill((float)nDigisPerChan[i * 36 + 34]);
  }
  if (meNDigisCHANBarrelCh36_) {
    for (int i = 0; i != 32; i++)
      if (nDigisPerChan[i * 36 + 35] > 0)
        meNDigisCHANBarrelCh36_->Fill((float)nDigisPerChan[i * 36 + 35]);
  }

  // Rate of events with >N digis:
  if (nEventDigis > bigEventSize) {
    if (bigEventRate)
      bigEventRate->Fill(lumiSection, 1. / 23.);
  }

  // Rate of pixel events and total number of pixel events per BX:
  if (nActiveModules >= 4) {
    if (pixEvtsPerBX)
      pixEvtsPerBX->Fill(float(bx));
    if (pixEventRate)
      pixEventRate->Fill(lumiSection, 1. / 23.);
  }

  if (slowDown)
    usleep(10000);
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelDigiSource::buildStructure(const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(trackerTopoTokenBeginRun_);
  const TrackerTopology* pTT = tTopoHandle.product();

  LogInfo("PixelDQM") << " SiPixelDigiSource::buildStructure";
  edm::ESHandle<TrackerGeometry> pDD = iSetup.getHandle(trackerGeomTokenBeginRun_);

  LogVerbatim("PixelDQM") << " *** Geometry node for TrackerGeom is  " << &(*pDD) << std::endl;
  LogVerbatim("PixelDQM") << " *** I have " << pDD->dets().size() << " detectors" << std::endl;
  LogVerbatim("PixelDQM") << " *** I have " << pDD->detTypes().size() << " types" << std::endl;

  for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) {
    if (dynamic_cast<PixelGeomDetUnit const*>((*it)) != nullptr) {
      DetId detId = (*it)->geographicalId();
      const GeomDetUnit* geoUnit = pDD->idToDetUnit(detId);
      const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
      int nrows = (pixDet->specificTopology()).nrows();
      int ncols = (pixDet->specificTopology()).ncolumns();

      if (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
        if (isPIB)
          continue;
        LogDebug("PixelDQM") << " ---> Adding Barrel Module " << detId.rawId() << endl;
        uint32_t id = detId();
        int layer = PixelBarrelName(DetId(id), pTT, isUpgrade).layerName();
        if (layer > noOfLayers)
          noOfLayers = layer;
        SiPixelDigiModule* theModule = new SiPixelDigiModule(id, ncols, nrows);
        thePixelStructure.insert(pair<uint32_t, SiPixelDigiModule*>(id, theModule));

      } else if ((detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) && (!isUpgrade)) {
        LogDebug("PixelDQM") << " ---> Adding Endcap Module " << detId.rawId() << endl;
        uint32_t id = detId();
        SiPixelDigiModule* theModule = new SiPixelDigiModule(id, ncols, nrows);

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

        thePixelStructure.insert(pair<uint32_t, SiPixelDigiModule*>(id, theModule));
      } else if ((detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) && (isUpgrade)) {
        LogDebug("PixelDQM") << " ---> Adding Endcap Module " << detId.rawId() << endl;
        uint32_t id = detId();
        SiPixelDigiModule* theModule = new SiPixelDigiModule(id, ncols, nrows);

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

        thePixelStructure.insert(pair<uint32_t, SiPixelDigiModule*>(id, theModule));
      }  //end_elseif(isUpgrade)
    }
  }
  LogInfo("PixelDQM") << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelDigiSource::bookMEs(DQMStore::IBooker& iBooker, const edm::EventSetup& iSetup) {
  // Get DQM interface
  iBooker.setCurrentFolder(topFolderName_);
  char title[80];
  sprintf(title, "Rate of events with >%i digis;LumiSection;Rate [Hz]", bigEventSize);
  bigEventRate = iBooker.book1D("bigEventRate", title, 5000, 0., 5000.);
  char title1[80];
  sprintf(title1, "Pixel events vs. BX;BX;# events");
  pixEvtsPerBX = iBooker.book1D("pixEvtsPerBX", title1, 3565, 0., 3565.);
  char title2[80];
  sprintf(title2, "Rate of Pixel events;LumiSection;Rate [Hz]");
  pixEventRate = iBooker.book1D("pixEventRate", title2, 5000, 0., 5000.);
  char title3[80];
  sprintf(title3, "Number of Zero-Occupancy Barrel ROCs;LumiSection;N_{ZERO-OCCUPANCY} Barrel ROCs");
  noOccROCsBarrel = iBooker.book1D("noOccROCsBarrel", title3, 500, 0., 5000.);
  char title4[80];
  sprintf(title4, "Number of Low-Efficiency Barrel ROCs;LumiSection;N_{LO EFF} Barrel ROCs");
  loOccROCsBarrel = iBooker.book1D("loOccROCsBarrel", title4, 500, 0., 5000.);
  char title5[80];
  sprintf(title5, "Number of Zero-Occupancy Endcap ROCs;LumiSection;N_{ZERO-OCCUPANCY} Endcap ROCs");
  noOccROCsEndcap = iBooker.book1D("noOccROCsEndcap", title5, 500, 0., 5000.);
  char title6[80];
  sprintf(title6, "Number of Low-Efficiency Endcap ROCs;LumiSection;N_{LO EFF} Endcap ROCs");
  loOccROCsEndcap = iBooker.book1D("loOccROCsEndcap", title6, 500, 0., 5000.);
  char title7[80];
  sprintf(title7, "Average digi occupancy per FED;FED;NDigis/<NDigis>");
  char title8[80];
  sprintf(title8, "FED Digi Occupancy (NDigis/<NDigis>) vs LumiSections;Lumi Section;FED");
  if (modOn) {
    {
      auto scope = DQMStore::IBooker::UseLumiScope(iBooker);
      averageDigiOccupancy = iBooker.bookProfile("averageDigiOccupancy", title7, 40, -0.5, 39.5, 0., 3.);
    }
    avgfedDigiOccvsLumi = iBooker.book2D("avgfedDigiOccvsLumi", title8, 640, 0., 3200., 40, -0.5, 39.5);
    avgBarrelFedOccvsLumi = iBooker.book1D(
        "avgBarrelFedOccvsLumi",
        "Average Barrel FED digi occupancy (<NDigis>) vs LumiSections;Lumi Section;Average digi occupancy per FED",
        320,
        0.,
        3200.);
    avgEndcapFedOccvsLumi = iBooker.book1D(
        "avgEndcapFedOccvsLumi",
        "Average Endcap FED digi occupancy (<NDigis>) vs LumiSections;Lumi Section;Average digi occupancy per FED",
        320,
        0.,
        3200.);
  }
  if (!modOn && !perLSsaving) {
    averageDigiOccupancy = iBooker.book1D(
        "averageDigiOccupancy", title7, 40, -0.5, 39.5);  //Book as TH1 for offline to ensure thread-safe behaviour
    avgfedDigiOccvsLumi = iBooker.book2D("avgfedDigiOccvsLumi", title8, 3200, 0., 3200., 40, -0.5, 39.5);
  }
  std::map<uint32_t, SiPixelDigiModule*>::iterator struct_iter;

  SiPixelFolderOrganizer theSiPixelFolder(false);

  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(trackerTopoTokenBeginRun_);
  const TrackerTopology* pTT = tTopoHandle.product();

  for (struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++) {
    /// Create folder tree and book histograms
    if (modOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 0, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 0, twoDimOn, hiRes, reducedSet, twoDimModOn, isUpgrade);
      } else {
        if (!isPIB)
          throw cms::Exception("LogicError") << "[SiPixelDigiSource::bookMEs] Creation of DQM folder failed";
      }
    }
    if (ladOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 1, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 1, twoDimOn, hiRes, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH LADDER-FOLDER\n";
      }
    }
    if (layOn || twoDimOnlyLayDisk) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 2, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 2, twoDimOn, hiRes, reducedSet, twoDimOnlyLayDisk, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH LAYER-FOLDER\n";
      }
    }

    if (phiOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 3, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 3, twoDimOn, hiRes, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH PHI-FOLDER\n";
      }
    }
    if (bladeOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 4, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 4, twoDimOn, hiRes, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH BLADE-FOLDER\n";
      }
    }
    if (diskOn || twoDimOnlyLayDisk) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 5, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 5, twoDimOn, hiRes, reducedSet, twoDimOnlyLayDisk, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH DISK-FOLDER\n";
      }
    }
    if (ringOn) {
      if (theSiPixelFolder.setModuleFolder(iBooker, (*struct_iter).first, 6, isUpgrade)) {
        (*struct_iter).second->book(conf_, pTT, iBooker, 6, twoDimOn, hiRes, reducedSet, isUpgrade);
      } else {
        LogDebug("PixelDQM") << "PROBLEM WITH RING-FOLDER\n";
      }
    }
  }
  iBooker.cd(topFolderName_ + "/Barrel");
  meNDigisCOMBBarrel_ = iBooker.book1D("ALLMODS_ndigisCOMB_Barrel", "Number of Digis", 200, 0., 400.);
  meNDigisCOMBBarrel_->setAxisTitle("Number of digis per module per event", 1);
  meNDigisCHANBarrel_ = iBooker.book1D("ALLMODS_ndigisCHAN_Barrel", "Number of Digis", 100, 0., 1000.);
  meNDigisCHANBarrel_->setAxisTitle("Number of digis per FED channel per event", 1);
  std::stringstream ss1, ss2;
  for (int i = 1; i <= noOfLayers; i++) {
    ss1.str(std::string());
    ss1 << "ALLMODS_ndigisCHAN_BarrelL" << i;
    ss2.str(std::string());
    ss2 << "Number of Digis L" << i;
    meNDigisCHANBarrelLs_.push_back(iBooker.book1D(ss1.str(), ss2.str(), 100, 0., 1000.));
    meNDigisCHANBarrelLs_.at(i - 1)->setAxisTitle("Number of digis per FED channel per event", 1);
  }
  meNDigisCHANBarrelCh1_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh1", "Number of Digis Ch1", 100, 0., 1000.);
  meNDigisCHANBarrelCh1_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh2_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh2", "Number of Digis Ch2", 100, 0., 1000.);
  meNDigisCHANBarrelCh2_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh3_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh3", "Number of Digis Ch3", 100, 0., 1000.);
  meNDigisCHANBarrelCh3_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh4_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh4", "Number of Digis Ch4", 100, 0., 1000.);
  meNDigisCHANBarrelCh4_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh5_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh5", "Number of Digis Ch5", 100, 0., 1000.);
  meNDigisCHANBarrelCh5_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh6_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh6", "Number of Digis Ch6", 100, 0., 1000.);
  meNDigisCHANBarrelCh6_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh7_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh7", "Number of Digis Ch7", 100, 0., 1000.);
  meNDigisCHANBarrelCh7_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh8_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh8", "Number of Digis Ch8", 100, 0., 1000.);
  meNDigisCHANBarrelCh8_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh9_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh9", "Number of Digis Ch9", 100, 0., 1000.);
  meNDigisCHANBarrelCh9_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh10_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh10", "Number of Digis Ch10", 100, 0., 1000.);
  meNDigisCHANBarrelCh10_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh11_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh11", "Number of Digis Ch11", 100, 0., 1000.);
  meNDigisCHANBarrelCh11_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh12_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh12", "Number of Digis Ch12", 100, 0., 1000.);
  meNDigisCHANBarrelCh12_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh13_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh13", "Number of Digis Ch13", 100, 0., 1000.);
  meNDigisCHANBarrelCh13_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh14_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh14", "Number of Digis Ch14", 100, 0., 1000.);
  meNDigisCHANBarrelCh14_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh15_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh15", "Number of Digis Ch15", 100, 0., 1000.);
  meNDigisCHANBarrelCh15_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh16_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh16", "Number of Digis Ch16", 100, 0., 1000.);
  meNDigisCHANBarrelCh16_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh17_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh17", "Number of Digis Ch17", 100, 0., 1000.);
  meNDigisCHANBarrelCh17_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh18_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh18", "Number of Digis Ch18", 100, 0., 1000.);
  meNDigisCHANBarrelCh18_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh19_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh19", "Number of Digis Ch19", 100, 0., 1000.);
  meNDigisCHANBarrelCh19_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh20_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh20", "Number of Digis Ch20", 100, 0., 1000.);
  meNDigisCHANBarrelCh20_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh21_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh21", "Number of Digis Ch21", 100, 0., 1000.);
  meNDigisCHANBarrelCh21_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh22_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh22", "Number of Digis Ch22", 100, 0., 1000.);
  meNDigisCHANBarrelCh22_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh23_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh23", "Number of Digis Ch23", 100, 0., 1000.);
  meNDigisCHANBarrelCh23_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh24_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh24", "Number of Digis Ch24", 100, 0., 1000.);
  meNDigisCHANBarrelCh24_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh25_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh25", "Number of Digis Ch25", 100, 0., 1000.);
  meNDigisCHANBarrelCh25_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh26_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh26", "Number of Digis Ch26", 100, 0., 1000.);
  meNDigisCHANBarrelCh26_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh27_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh27", "Number of Digis Ch27", 100, 0., 1000.);
  meNDigisCHANBarrelCh27_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh28_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh28", "Number of Digis Ch28", 100, 0., 1000.);
  meNDigisCHANBarrelCh28_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh29_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh29", "Number of Digis Ch29", 100, 0., 1000.);
  meNDigisCHANBarrelCh29_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh30_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh30", "Number of Digis Ch30", 100, 0., 1000.);
  meNDigisCHANBarrelCh30_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh31_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh31", "Number of Digis Ch31", 100, 0., 1000.);
  meNDigisCHANBarrelCh31_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh32_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh32", "Number of Digis Ch32", 100, 0., 1000.);
  meNDigisCHANBarrelCh32_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh33_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh33", "Number of Digis Ch33", 100, 0., 1000.);
  meNDigisCHANBarrelCh33_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh34_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh34", "Number of Digis Ch34", 100, 0., 1000.);
  meNDigisCHANBarrelCh34_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh35_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh35", "Number of Digis Ch35", 100, 0., 1000.);
  meNDigisCHANBarrelCh35_->setAxisTitle("Number of digis per FED channel per event", 1);
  meNDigisCHANBarrelCh36_ = iBooker.book1D("ALLMODS_ndigisCHAN_BarrelCh36", "Number of Digis Ch36", 100, 0., 1000.);
  meNDigisCHANBarrelCh36_->setAxisTitle("Number of digis per FED channel per event", 1);
  iBooker.cd(topFolderName_ + "/Endcap");
  meNDigisCOMBEndcap_ = iBooker.book1D("ALLMODS_ndigisCOMB_Endcap", "Number of Digis", 200, 0., 400.);
  meNDigisCOMBEndcap_->setAxisTitle("Number of digis per module per event", 1);
  meNDigisCHANEndcap_ = iBooker.book1D("ALLMODS_ndigisCHAN_Endcap", "Number of Digis", 100, 0., 1000.);
  meNDigisCHANEndcap_->setAxisTitle("Number of digis per FED channel per event", 1);
  for (int i = 1; i <= noOfDisks; i++) {
    ss1.str(std::string());
    ss1 << "ALLMODS_ndigisCHAN_EndcapDp" << i;
    ss2.str(std::string());
    ss2 << "Number of Digis Disk p" << i;
    meNDigisCHANEndcapDps_.push_back(iBooker.book1D(ss1.str(), ss2.str(), 100, 0., 1000.));
    meNDigisCHANEndcapDps_.at(i - 1)->setAxisTitle("Number of digis per FED channel per event", 1);
  }
  for (int i = 1; i <= noOfDisks; i++) {
    ss1.str(std::string());
    ss1 << "ALLMODS_ndigisCHAN_EndcapDm" << i;
    ss2.str(std::string());
    ss2 << "Number of Digis Disk m" << i;
    meNDigisCHANEndcapDms_.push_back(iBooker.book1D(ss1.str(), ss2.str(), 100, 0., 1000.));
    meNDigisCHANEndcapDms_.at(i - 1)->setAxisTitle("Number of digis per FED channel per event", 1);
  }
  iBooker.cd(topFolderName_);
}

void SiPixelDigiSource::CountZeroROCsInSubstructure(bool barrel, bool& DoZeroRocs, SiPixelDigiModule* mod) {
  std::pair<int, int> tempPair = mod->getZeroLoEffROCs();

  if (barrel) {
    NzeroROCs[0] += tempPair.first;
    NloEffROCs[0] += tempPair.second;
  } else {
    NzeroROCs[1] += tempPair.first;
    NloEffROCs[1] += tempPair.second;
  }

  DoZeroRocs = false;
  mod->resetRocMap();  //once got the number of ZeroOccupancy Rocs, reset the ROC map of the corresponding Pixel substructure
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelDigiSource);
