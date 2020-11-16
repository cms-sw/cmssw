#include "CondTools/SiStrip/plugins/SiStripFedCablingBuilder.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"
#include <iostream>
#include <fstream>
#include <sstream>

// -----------------------------------------------------------------------------
//
SiStripFedCablingBuilder::SiStripFedCablingBuilder(const edm::ParameterSet& pset)
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
void SiStripFedCablingBuilder::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  edm::LogInfo("SiStripFedCablingBuilder")
      << "... creating dummy SiStripFedCabling Data for Run " << run.run() << "\n " << std::endl;

  edm::LogVerbatim("SiStripFedCablingBuilder") << "[SiStripFedCablingBuilder::" << __func__ << "]"
                                               << " Retrieving FED cabling...";
  auto fed = setup.getHandle(fedCablingToken_);

  edm::LogVerbatim("SiStripFedCablingBuilder") << "[SiStripFedCablingBuilder::" << __func__ << "]"
                                               << " Retrieving FEC cabling...";
  auto fec = setup.getHandle(fecCablingToken_);

  edm::LogVerbatim("SiStripFedCablingBuilder") << "[SiStripFedCablingBuilder::" << __func__ << "]"
                                               << " Retrieving DET cabling...";
  auto det = setup.getHandle(detCablingToken_);

  edm::LogVerbatim("SiStripFedCablingBuilder") << "[SiStripFedCablingBuilder::" << __func__ << "]"
                                               << " Retrieving REGION cabling...";
  auto region = setup.getHandle(regionCablingToken_);

  if (!fed.isValid()) {
    edm::LogError("SiStripFedCablingBuilder") << " Invalid handle to FED cabling object: ";
    return;
  }

  const auto tTopo = &setup.getData(tTopoToken_);
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingBuilder::" << __func__ << "]"
       << " VERBOSE DEBUG" << std::endl;
    fed->print(ss, tTopo);
    ss << std::endl;
    if (printFecCabling_ && fec.isValid()) {
      fec->print(ss);
    }
    ss << std::endl;
    if (printDetCabling_ && det.isValid()) {
      det->print(ss);
    }
    ss << std::endl;
    if (printRegionCabling_ && region.isValid()) {
      region->print(ss);
    }
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingBuilder") << ss.str();
  }

  {
    std::stringstream ss;
    ss << "[SiStripFedCablingBuilder::" << __func__ << "]"
       << " TERSE DEBUG" << std::endl;
    fed->terse(ss);
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingBuilder") << ss.str();
  }

  {
    std::stringstream ss;
    ss << "[SiStripFedCablingBuilder::" << __func__ << "]"
       << " SUMMARY DEBUG" << std::endl;
    fed->summary(ss, tTopo);
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingBuilder") << ss.str();
  }

  edm::LogVerbatim("SiStripFedCablingBuilder") << "[SiStripFedCablingBuilder::" << __func__ << "]"
                                               << " Copying FED cabling...";
  SiStripFedCabling* obj = new SiStripFedCabling(*(fed.product()));

  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest("SiStripFedCablingRcd")) {
      mydbservice->createNewIOV<SiStripFedCabling>(
          obj, mydbservice->beginOfTime(), mydbservice->endOfTime(), "SiStripFedCablingRcd");
    } else {
      mydbservice->appendSinceTime<SiStripFedCabling>(obj, mydbservice->currentTime(), "SiStripFedCablingRcd");
    }
  } else {
    edm::LogError("SiStripFedCablingBuilder") << "Service is unavailable" << std::endl;
  }
}
