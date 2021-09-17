#ifndef SiPixelVCalReader_H
#define SiPixelVCalReader_H

#include <stdio.h>
#include <sys/time.h>
#include <iomanip>  // std::setw
#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "TFile.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TTree.h"
//#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
//#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
//#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/SiPixelVCalRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelVCalSimRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"
#include "CondTools/SiPixel/test/SiPixelVCalDB.h"
//#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class SiPixelVCalReader : public edm::EDAnalyzer {
public:
  explicit SiPixelVCalReader(const edm::ParameterSet&);
  ~SiPixelVCalReader();
  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  bool printdebug_;
  bool useSimRcd_;
  TH1F* slopeBPix_;
  TH1F* slopeFPix_;
  TH1F* offsetBPix_;
  TH1F* offsetFPix_;
};

#endif
