// system includes
#include <iostream>
#include <sstream>

// user includes
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripFecCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripFedCablingReader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  SiStripFedCablingReader(const edm::ParameterSet&);
  ~SiStripFedCablingReader() override = default;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override{};
  void analyze(const edm::Event&, const edm::EventSetup&) override{};

private:
  bool printFecCabling_;
  bool printDetCabling_;
  bool printRegionCabling_;
  const edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;
  const edm::ESGetToken<SiStripFecCabling, SiStripFecCablingRcd> fecCablingToken_;
  const edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;
  const edm::ESGetToken<SiStripRegionCabling, SiStripRegionCablingRcd> regionCablingToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
};

// -----------------------------------------------------------------------------
//
SiStripFedCablingReader::SiStripFedCablingReader(const edm::ParameterSet& pset)
    : printFecCabling_(pset.getUntrackedParameter<bool>("PrintFecCabling", false)),
      printDetCabling_(pset.getUntrackedParameter<bool>("PrintDetCabling", false)),
      printRegionCabling_(pset.getUntrackedParameter<bool>("PrintRegionCabling", false)),
      fedCablingToken_(esConsumes<edm::Transition::BeginRun>()),
      fecCablingToken_(esConsumes<edm::Transition::BeginRun>()),
      detCablingToken_(esConsumes<edm::Transition::BeginRun>()),
      regionCablingToken_(esConsumes<edm::Transition::BeginRun>()),
      tTopoToken_(esConsumes<edm::Transition::BeginRun>()) {}

// -----------------------------------------------------------------------------
//
void SiStripFedCablingReader::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  auto const fedRec = setup.tryToGet<SiStripFedCablingRcd>();
  auto const fecRec = setup.tryToGet<SiStripFecCablingRcd>();
  auto const detRec = setup.tryToGet<SiStripDetCablingRcd>();
  auto const regRec = setup.tryToGet<SiStripRegionCablingRcd>();

  edm::ESHandle<SiStripFedCabling> fed;
  if (fedRec) {
    edm::LogVerbatim("SiStripFedCablingReader") << "[SiStripFedCablingReader::" << __func__ << "]"
                                                << " Retrieving FED cabling...";
    fed = setup.getHandle(fedCablingToken_);
  }

  edm::ESHandle<SiStripFecCabling> fec;
  if (fecRec) {
    edm::LogVerbatim("SiStripFedCablingReader") << "[SiStripFedCablingReader::" << __func__ << "]"
                                                << " Retrieving FEC cabling...";
    fec = setup.getHandle(fecCablingToken_);
  }

  edm::ESHandle<SiStripDetCabling> det;
  if (detRec) {
    edm::LogVerbatim("SiStripFedCablingReader") << "[SiStripFedCablingReader::" << __func__ << "]"
                                                << " Retrieving DET cabling...";
    det = setup.getHandle(detCablingToken_);
  }

  edm::ESHandle<SiStripRegionCabling> region;
  if (regRec) {
    edm::LogVerbatim("SiStripFedCablingReader") << "[SiStripFedCablingReader::" << __func__ << "]"
                                                << " Retrieving REGION cabling...";
    region = setup.getHandle(regionCablingToken_);
  }

  if (!fed.isValid()) {
    edm::LogError("SiStripFedCablingReader") << " Invalid handle to FED cabling object: ";
    return;
  }

  {
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " VERBOSE DEBUG" << std::endl;
    if (fedRec) {
      fed->print(ss, &setup.getData(tTopoToken_));
    }
    ss << std::endl;
    if (fecRec && printFecCabling_ && fec.isValid()) {
      fec->print(ss);
    }
    ss << std::endl;
    if (detRec && printDetCabling_ && det.isValid()) {
      det->print(ss);
    }
    ss << std::endl;
    if (regRec && printRegionCabling_ && region.isValid()) {
      region->print(ss);
    }
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }

  if (fedRec) {
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " TERSE DEBUG" << std::endl;
    fed->terse(ss);
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }

  if (fedRec) {
    std::stringstream ss;
    ss << "[SiStripFedCablingReader::" << __func__ << "]"
       << " SUMMARY DEBUG" << std::endl;
    fed->summary(ss, &setup.getData(tTopoToken_));
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingReader") << ss.str();
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripFedCablingReader);
