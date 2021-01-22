#ifndef SiStripFedCablingBuilder_H
#define SiStripFedCablingBuilder_H
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripFecCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"

class SiStripFedCabling;
class SiStripFecCabling;
class SiStripDetCabling;
class SiStripRegionCabling;

class SiStripFedCablingBuilder : public edm::EDAnalyzer {
public:
  SiStripFedCablingBuilder(const edm::ParameterSet& iConfig);

  ~SiStripFedCablingBuilder() override{};

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override { ; }

private:
  bool printFecCabling_;
  bool printDetCabling_;
  bool printRegionCabling_;
  edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;
  edm::ESGetToken<SiStripFecCabling, SiStripFecCablingRcd> fecCablingToken_;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;
  edm::ESGetToken<SiStripRegionCabling, SiStripRegionCablingRcd> regionCablingToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
};
#endif
