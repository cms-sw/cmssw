#ifndef SiPixelVCalReader_H
#define SiPixelVCalReader_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "TFile.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TTree.h"
#include <cstdio>
#include <iomanip>  // std::setw
#include <iostream>
#include <sys/time.h>
#include "SiPixelVCalDB.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/SiPixelVCalRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelVCalSimRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"

class SiPixelVCalReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiPixelVCalReader(const edm::ParameterSet&);
  ~SiPixelVCalReader() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<SiPixelVCal, SiPixelVCalSimRcd> siPixelVCalSimToken_;
  const edm::ESGetToken<SiPixelVCal, SiPixelVCalRcd> siPixelVCalToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken_;

  bool printdebug_;
  bool useSimRcd_;

  TH1F* slopeBPix_;
  TH1F* slopeFPix_;
  TH1F* offsetBPix_;
  TH1F* offsetFPix_;
};

#endif
