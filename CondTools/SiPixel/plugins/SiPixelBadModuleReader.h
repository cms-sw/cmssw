#ifndef CondTools_SiPixel_SiPixelBadModuleReader_H
#define CondTools_SiPixel_SiPixelBadModuleReader_H

// system include files

// user include files
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

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

  uint32_t printdebug_;
  std::string whichRcd;
  TH2F *_TH2F_dead_modules_BPIX_lay1;
  TH2F *_TH2F_dead_modules_BPIX_lay2;
  TH2F *_TH2F_dead_modules_BPIX_lay3;
  TH2F *_TH2F_dead_modules_FPIX_minusZ_disk1;
  TH2F *_TH2F_dead_modules_FPIX_minusZ_disk2;
  TH2F *_TH2F_dead_modules_FPIX_plusZ_disk1;
  TH2F *_TH2F_dead_modules_FPIX_plusZ_disk2;
};
#endif
