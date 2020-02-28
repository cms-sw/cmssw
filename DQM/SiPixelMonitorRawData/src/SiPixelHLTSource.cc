// -*- C++ -*-
//
// Package:    SiPixelMonitorRawData
// Class:      SiPixelHLTSource
//
/**\class

 Description:
 Produces histograms for error information generated at the raw2digi stage for
 the pixel tracker.

 Implementation:
 Takes raw data and error data as input, and uses it to populate three
 histograms indexed by FED id.

*/
//
// Original Author:  Andrew York
//
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
// DQM Framework
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
#include "DQM/SiPixelMonitorRawData/interface/SiPixelHLTSource.h"
#include "DQMServices/Core/interface/DQMStore.h"
// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// DataFormats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
//
#include <cstdlib>
#include <string>

using namespace std;
using namespace edm;

SiPixelHLTSource::SiPixelHLTSource(const edm::ParameterSet &iConfig)
    : conf_(iConfig),
      rawin_(consumes<FEDRawDataCollection>(conf_.getParameter<edm::InputTag>("RawInput"))),
      errin_(consumes<edm::DetSetVector<SiPixelRawDataError>>(conf_.getParameter<edm::InputTag>("ErrorInput"))),
      saveFile(conf_.getUntrackedParameter<bool>("saveFile", false)),
      slowDown(conf_.getUntrackedParameter<bool>("slowDown", false)),
      dirName_(conf_.getUntrackedParameter<string>("DirName", "Pixel/FEDIntegrity/")) {
  firstRun = true;
  LogInfo("PixelDQM") << "SiPixelHLTSource::SiPixelHLTSource: Got DQM BackEnd interface" << endl;
}

SiPixelHLTSource::~SiPixelHLTSource() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  LogInfo("PixelDQM") << "SiPixelHLTSource::~SiPixelHLTSource: Destructor" << endl;
}

void SiPixelHLTSource::dqmBeginRun(const edm::Run &r, const edm::EventSetup &iSetup) {
  LogInfo("PixelDQM") << " SiPixelHLTSource::beginJob - Initialisation ... " << std::endl;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  if (firstRun) {
    eventNo = 0;

    firstRun = false;
  }
}

void SiPixelHLTSource::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, edm::EventSetup const &) {
  // Book Monitoring Elements
  bookMEs(iBooker);
}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelHLTSource::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  eventNo++;
  // get raw input data
  edm::Handle<FEDRawDataCollection> rawinput;
  iEvent.getByToken(rawin_, rawinput);
  // get error input data
  edm::Handle<edm::DetSetVector<SiPixelRawDataError>> errorinput;
  iEvent.getByToken(errin_, errorinput);
  if (!errorinput.isValid())
    return;

  int fedId;

  for (fedId = 0; fedId <= 39; fedId++) {
    // get event data for this fed
    const FEDRawData &fedRawData = rawinput->FEDData(fedId);
    if (fedRawData.size() != 0)
      (meRawWords_)->Fill(fedId);
  }  // end for

  edm::DetSet<SiPixelRawDataError>::const_iterator di;

  for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) {
    if (GeomDetEnumerators::isTrackerPixel((*it)->subDetector())) {
      uint32_t detId = (*it)->geographicalId();
      edm::DetSetVector<SiPixelRawDataError>::const_iterator isearch = errorinput->find(detId);
      if (isearch != errorinput->end()) {
        for (di = isearch->data.begin(); di != isearch->data.end(); di++) {
          fedId = di->getFedId();         // FED the error came from
          int errorType = di->getType();  // type of error
          switch (errorType) {
            case (35):
              (meNErrors_)->Fill(fedId);
              break;
            case (36):
              (meNErrors_)->Fill(fedId);
              break;
            case (37):
              (meNErrors_)->Fill(fedId);
              break;
            case (38):
              (meNErrors_)->Fill(fedId);
              break;
            default:
              break;
          };  // end switch
        }     // end for(di
      }       // end if( isearch
    }         // end if( ((*it)->subDetector()
  }           // for(TrackerGeometry

  edm::DetSetVector<SiPixelRawDataError>::const_iterator isearch = errorinput->find(0xffffffff);

  if (isearch != errorinput->end()) {  // Not at empty iterator
    for (di = isearch->data.begin(); di != isearch->data.end(); di++) {
      fedId = di->getFedId();         // FED the error came from
      int errorType = di->getType();  // type of error
      switch (errorType) {
        case (35):
          (meNErrors_)->Fill(fedId);
          break;
        case (36):
          (meNErrors_)->Fill(fedId);
          break;
        case (37):
          (meNErrors_)->Fill(fedId);
          break;
        case (38):
          (meNErrors_)->Fill(fedId);
          break;
        case (39):
          (meNCRCs_)->Fill(fedId);
          break;
        default:
          break;
      };  // end switch
    }     // end for(di
  }       // end if( isearch
  // slow down...
  if (slowDown)
    usleep(100000);
}

//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelHLTSource::bookMEs(DQMStore::IBooker &iBooker) {
  iBooker.cd();
  iBooker.setCurrentFolder(dirName_);

  std::string rawhid;
  std::string errhid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag rawin = conf_.getParameter<edm::InputTag>("RawInput");
  SiPixelHistogramId *RawHistogramId = new SiPixelHistogramId(rawin.label());
  edm::InputTag errin = conf_.getParameter<edm::InputTag>("ErrorInput");
  SiPixelHistogramId *ErrorHistogramId = new SiPixelHistogramId(errin.label());

  // Is a FED sending raw data
  meRawWords_ = iBooker.book1D("FEDEntries", "Number of raw words", 40, -0.5, 39.5);
  meRawWords_->setAxisTitle("Number of raw words", 1);

  // Number of CRC errors
  meNCRCs_ = iBooker.book1D("FEDFatal", "Number of fatal errors", 40, -0.5, 39.5);
  meNCRCs_->setAxisTitle("Number of fatal errors", 1);

  // Number of translation error words
  meNErrors_ = iBooker.book1D("FEDNonFatal", "Number of non-fatal errors", 40, -0.5, 39.5);
  meNErrors_->setAxisTitle("Number of non-fatal errors", 1);

  delete RawHistogramId;
  delete ErrorHistogramId;
}

DEFINE_FWK_MODULE(SiPixelHLTSource);
