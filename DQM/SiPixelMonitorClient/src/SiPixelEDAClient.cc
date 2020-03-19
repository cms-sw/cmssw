#include "DQM/SiPixelMonitorClient/interface/SiPixelEDAClient.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelDataQuality.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelInformationExtractor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#define BUF_SIZE 256

using namespace edm;
using namespace std;
//
// -- Constructor
//
SiPixelEDAClient::SiPixelEDAClient(const edm::ParameterSet &ps) {
  // cout<<"Entering  SiPixelEDAClient::SiPixelEDAClient: "<<endl;

  edm::LogInfo("SiPixelEDAClient") << " Creating SiPixelEDAClient "
                                   << "\n";

  summaryFrequency_ = ps.getUntrackedParameter<int>("SummaryCreationFrequency", 20);
  tkMapFrequency_ = ps.getUntrackedParameter<int>("TkMapCreationFrequency", 50);
  staticUpdateFrequency_ = ps.getUntrackedParameter<int>("StaticUpdateFrequency", 10);
  actionOnLumiSec_ = ps.getUntrackedParameter<bool>("ActionOnLumiSection", false);  // client
  actionOnRunEnd_ = ps.getUntrackedParameter<bool>("ActionOnRunEnd", true);         // client
  evtOffsetForInit_ = ps.getUntrackedParameter<int>("EventOffsetForInit", 10);      // client
  offlineXMLfile_ = ps.getUntrackedParameter<bool>("UseOfflineXMLFile", false);     // client
  hiRes_ = ps.getUntrackedParameter<bool>("HighResolutionOccupancy",
                                          false);                                               // client
  noiseRate_ = ps.getUntrackedParameter<double>("NoiseRateCutValue", 0.001);                    // client
  noiseRateDenominator_ = ps.getUntrackedParameter<int>("NEventsForNoiseCalculation", 100000);  // client
  Tier0Flag_ = ps.getUntrackedParameter<bool>("Tier0Flag", false);                              // client
  doHitEfficiency_ = ps.getUntrackedParameter<bool>("DoHitEfficiency", true);                   // client
  inputSource_ = ps.getUntrackedParameter<string>("inputSource", "source");
  isUpgrade_ = ps.getUntrackedParameter<bool>("isUpgrade", false);  // client

  if (!Tier0Flag_) {
    string localPath = string("DQM/SiPixelMonitorClient/test/loader.html");
    std::ifstream fin(edm::FileInPath(localPath).fullPath().c_str(), ios::in);
    char buf[BUF_SIZE];

    if (!fin) {
      cerr << "Input File: loader.html"
           << " could not be opened!" << endl;
      return;
    }

    while (fin.getline(buf, BUF_SIZE, '\n')) {  // pops off the newline character
      html_out_ << buf;
    }
    fin.close();
  }

  firstLumi = true;

  // instantiate the three work horses of the client:
  sipixelInformationExtractor_ = new SiPixelInformationExtractor(offlineXMLfile_);
  sipixelActionExecutor_ = new SiPixelActionExecutor(offlineXMLfile_, Tier0Flag_);
  sipixelDataQuality_ = new SiPixelDataQuality(offlineXMLfile_);

  inputSourceToken_ = consumes<FEDRawDataCollection>(ps.getUntrackedParameter<string>("inputSource", "source"));
  // cout<<"...leaving  SiPixelEDAClient::SiPixelEDAClient. "<<endl;
}

//
// -- Destructor
//
SiPixelEDAClient::~SiPixelEDAClient() {
  //  cout<<"Entering SiPixelEDAClient::~SiPixelEDAClient: "<<endl;

  edm::LogInfo("SiPixelEDAClient") << " Deleting SiPixelEDAClient "
                                   << "\n";

  if (sipixelInformationExtractor_) {
    delete sipixelInformationExtractor_;
    sipixelInformationExtractor_ = nullptr;
  }
  if (sipixelActionExecutor_) {
    delete sipixelActionExecutor_;
    sipixelActionExecutor_ = nullptr;
  }
  if (sipixelDataQuality_) {
    delete sipixelDataQuality_;
    sipixelDataQuality_ = nullptr;
  }

  //  cout<<"...leaving SiPixelEDAClient::~SiPixelEDAClient. "<<endl;
}
//
// -- Begin Run
//
void SiPixelEDAClient::beginRun(Run const &run, edm::EventSetup const &eSetup) {
  edm::LogInfo("SiPixelEDAClient") << "[SiPixelEDAClient]: Begining of Run";
  //  cout<<"Entering SiPixelEDAClient::beginRun: "<<endl;

  /// cout << "-----------NEW RUN---------------" << endl;

  if (firstLumi) {
    summaryFrequency_ = -1;
    tkMapFrequency_ = -1;
    actionOnRunEnd_ = true;
    evtOffsetForInit_ = -1;

    nLumiSecs_ = 0;
    nEvents_ = 0;
    if (Tier0Flag_)
      nFEDs_ = 40;
    else
      nFEDs_ = 0;
  }

  //  cout<<"...leaving SiPixelEDAClient::beginRun. "<<endl;
}

//
// -- End Luminosity Block
//
void SiPixelEDAClient::dqmEndLuminosityBlock(DQMStore::IBooker &iBooker,
                                             DQMStore::IGetter &iGetter,
                                             edm::LuminosityBlock const &lumiSeg,
                                             edm::EventSetup const &eSetup) {
  // cout<<"Entering SiPixelEDAClient::endLuminosityBlock: "<<endl;

  edm::LogInfo("SiPixelEDAClient") << "[SiPixelEDAClient]: Begin of LS transition";

  // Moved from beginLumi
  nEvents_lastLS_ = 0;
  nErrorsBarrel_lastLS_ = 0;
  nErrorsEndcap_lastLS_ = 0;
  MonitorElement *me = iGetter.get("Pixel/AdditionalPixelErrors/byLumiErrors");
  if (me) {
    nEvents_lastLS_ = int(me->getBinContent(0));
    nErrorsBarrel_lastLS_ = int(me->getBinContent(1));
    nErrorsEndcap_lastLS_ = int(me->getBinContent(2));
    me->Reset();
  }

  /// std::cout << "CREATING SUMMARY" << std::endl;
  sipixelActionExecutor_->createSummary(iBooker, iGetter, isUpgrade_);

  if (firstLumi) {
    iBooker.setCurrentFolder("Pixel/");
    iGetter.setCurrentFolder("Pixel/");
    // Creating Summary Histos:
    // std::cout << "CREATING SUMMARY" << std::endl;
    // sipixelActionExecutor_->createSummary(iBooker,iGetter, isUpgrade_);
    // Booking Deviation Histos:
    if (!Tier0Flag_)
      sipixelActionExecutor_->bookDeviations(iBooker, isUpgrade_);
    // Booking Efficiency Histos:
    if (doHitEfficiency_)
      sipixelActionExecutor_->bookEfficiency(iBooker, isUpgrade_);
    // Creating occupancy plots:
    sipixelActionExecutor_->bookOccupancyPlots(iBooker, iGetter, hiRes_);
    // Booking noisy pixel ME's:
    if (noiseRate_ > 0.)
      sipixelInformationExtractor_->bookNoisyPixels(iBooker, noiseRate_, Tier0Flag_);
    // Booking summary report ME's:
    sipixelDataQuality_->bookGlobalQualityFlag(iBooker, Tier0Flag_, nFEDs_);

    if (!Tier0Flag_) {
      MonitorElement *mefed = iGetter.get("Pixel/EventInfo/DAQContents/fedcounter");
      if (mefed) {
        for (int i = 0; i < mefed->getNbinsX(); ++i)
          nFEDs_ += mefed->getBinContent(i + 1);
      }
    }
    eSetup.get<SiPixelFedCablingMapRcd>().get(theCablingMap);

    firstLumi = false;
  }

  edm::LogInfo("SiPixelEDAClient") << "[SiPixelEDAClient]: End of LS transition, performing the DQM client "
                                      "operation";
  //
  nLumiSecs_ = lumiSeg.id().luminosityBlock();

  /// std::cout << "NEW LUMISECTION n " << nLumiSecs_ << std::endl;
  // nLumiSecs_++;

  edm::LogInfo("SiPixelEDAClient") << "====================================================== " << endl
                                   << " ===> Iteration # " << nLumiSecs_ << " " << lumiSeg.luminosityBlock() << endl
                                   << "====================================================== " << endl;

  if (Tier0Flag_)
    sipixelActionExecutor_->normaliseAvDigiOccVsLumi(iBooker, iGetter, nLumiSecs_);

  bool init = true;
  if (actionOnLumiSec_ && nLumiSecs_ % 1 == 0) {
    if (doHitEfficiency_)
      sipixelActionExecutor_->createEfficiency(iBooker, iGetter, isUpgrade_);
    sipixelActionExecutor_->createOccupancy(iBooker, iGetter);
    iBooker.cd();
    iGetter.cd();
    sipixelDataQuality_->computeGlobalQualityFlagByLumi(
        iGetter, init, nFEDs_, Tier0Flag_, nEvents_lastLS_, nErrorsBarrel_lastLS_, nErrorsEndcap_lastLS_);
    init = true;
    iBooker.cd();
    iGetter.cd();
    sipixelDataQuality_->fillGlobalQualityPlot(iBooker, iGetter, init, theCablingMap, nFEDs_, Tier0Flag_, nLumiSecs_);
    init = true;
    if (noiseRate_ >= 0.)
      sipixelInformationExtractor_->findNoisyPixels(
          iBooker, iGetter, init, noiseRate_, noiseRateDenominator_, theCablingMap);
  }

  // cout<<"...leaving SiPixelEDAClient::endLuminosityBlock. "<<endl;
}
//
// -- End Job
//
void SiPixelEDAClient::dqmEndJob(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter) {
  //  cout<<"In SiPixelEDAClient::endJob "<<endl;
  edm::LogInfo("SiPixelEDAClient") << "[SiPixelEDAClient]: endjob called!";
  /// cout << "[SiPixelEDAClient]: endjob called!" << endl;
  sipixelActionExecutor_->createSummary(iBooker, iGetter, isUpgrade_);

  if (actionOnRunEnd_) {
    // sipixelActionExecutor_->createSummary(iBooker, iGetter, isUpgrade_);

    if (doHitEfficiency_) {
      sipixelActionExecutor_->createEfficiency(iBooker, iGetter, isUpgrade_);
      sipixelActionExecutor_->fillEfficiencySummary(iBooker, iGetter);
    }

    sipixelActionExecutor_->createOccupancy(iBooker, iGetter);

    if (Tier0Flag_)
      sipixelActionExecutor_->normaliseAvDigiOcc(iBooker, iGetter);

    iBooker.cd();
    iGetter.cd();
    bool init = true;
    sipixelDataQuality_->computeGlobalQualityFlag(iBooker, iGetter, init, nFEDs_, Tier0Flag_);
    init = true;
    iBooker.cd();
    iGetter.cd();

    sipixelDataQuality_->fillGlobalQualityPlot(iBooker, iGetter, init, theCablingMap, nFEDs_, Tier0Flag_, nLumiSecs_);
    init = true;
    if (noiseRate_ >= 0.)
      sipixelInformationExtractor_->findNoisyPixels(
          iBooker, iGetter, init, noiseRate_, noiseRateDenominator_, theCablingMap);
  }
}
