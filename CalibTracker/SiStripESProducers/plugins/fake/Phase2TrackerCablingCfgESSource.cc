#include "CalibTracker/SiStripESProducers/plugins/fake/Phase2TrackerCablingCfgESSource.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripModule.h"
#include "CalibTracker/Records/interface/SiStripHashedDetIdRcd.h"
#include "CalibTracker/SiStripCommon/interface/SiStripFedIdListReader.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>
#include <vector>
#include <map>

// -----------------------------------------------------------------------------
//
Phase2TrackerCablingCfgESSource::Phase2TrackerCablingCfgESSource( const edm::ParameterSet& pset )
  : Phase2TrackerCablingESProducer( pset ), pset_(pset)
{
  findingRecord<Phase2TrackerCablingRcd>();
  edm::LogVerbatim("Phase2TrackerCabling") 
    << "[Phase2TrackerCablingCfgESSource::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
Phase2TrackerCablingCfgESSource::~Phase2TrackerCablingCfgESSource() {
  edm::LogVerbatim("Phase2TrackerCabling")
    << "[Phase2TrackerCablingCfgESSource::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
Phase2TrackerCabling* Phase2TrackerCablingCfgESSource::make( const Phase2TrackerCablingRcd& ) {
  edm::LogVerbatim("Phase2TrackerCabling")
    << "[Phase2TrackerCablingCfgESSource::" << __func__ << "]"
    << " Building FED cabling map from cfg.";
  
  std::vector<Phase2TrackerModule> conns;

  // iterate through the parameterset and create corresponding Phase2TrackerModule
  std::vector<edm::ParameterSet> modules = pset_.getParameterSetVector("modules");
  uint32_t detid,gbtid,fedid,fedch,powerGroup,coolingLoop;
  for(std::vector<edm::ParameterSet>::const_iterator it = modules.begin();it<modules.end();++it) {
     detid = it->getParameter<uint32_t>("detid");
     gbtid = it->getParameter<uint32_t>("gbtid");
     fedid = it->getParameter<uint32_t>("fedid");
     fedch = it->getParameter<uint32_t>("fedch");
     powerGroup = it->getParameter<uint32_t>("powerGroup");
     coolingLoop = it->getParameter<uint32_t>("coolingLoop");
     Phase2TrackerModule::ModuleTypes type = it->getParameter<std::string>("moduleType")=="2S" ? Phase2TrackerModule::SS : Phase2TrackerModule::PS;
     conns.push_back(Phase2TrackerModule(type,detid,gbtid,fedid,fedch,powerGroup,coolingLoop));
  }
  
  // return the cabling 
  Phase2TrackerCabling* cabling = new Phase2TrackerCabling( conns );
  return cabling;
  
}

