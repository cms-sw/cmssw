#ifndef CondTools_SiStrip_FedCablingReader_H
#define CondTools_SiStrip_FedCablingReader_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripFecCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"

class SiStripFedCabling;
class SiStripFecCabling;
class SiStripDetCabling;
class SiStripRegionCabling;

class SiStripFedCablingReader : public edm::EDAnalyzer {
public:
  SiStripFedCablingReader(const edm::ParameterSet&);

  ~SiStripFedCablingReader() override { ; }

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

#endif  // CondTools_SiStrip_FedCablingReader_H
