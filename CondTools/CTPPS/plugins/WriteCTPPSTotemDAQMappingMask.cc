/****************************************************************************
*
* Offline analyzer for writing TOTEM DAQ Mapping sqlite file 
*
****************************************************************************/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondCore/CondDB/interface/Time.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/Utilities/interface/ESInputTag.h"
#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"
#include "CondFormats/PPSObjects/interface/TotemAnalysisMask.h"
#include "CondFormats/DataRecord/interface/TotemAnalysisMaskRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cstdint>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Prints ant writes to SQLite the DAQ mapping loaded by TotemDAQMappingESSourceXML.
 **/
class WriteCTPPSTotemDAQMappingMask: public edm::one::EDAnalyzer<> {
public:
  WriteCTPPSTotemDAQMappingMask(const edm::ParameterSet &ps);
  ~WriteCTPPSTotemDAQMappingMask() override {}

private:  
  cond::Time_t daqmappingiov;
  std::string record_map;
  std::string record_mask;
  std::string label;
  edm::ESGetToken<TotemDAQMapping, TotemReadoutRcd> tokenMapping;
  edm::ESGetToken<TotemAnalysisMask, TotemReadoutRcd> tokenAnalysisMask;

  void analyze(const edm::Event &e, const edm::EventSetup &es) override;
  void printMapping(TotemDAQMapping mapping);
  void printMask(TotemAnalysisMask analysisMask);
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

WriteCTPPSTotemDAQMappingMask::WriteCTPPSTotemDAQMappingMask(const edm::ParameterSet &ps): 
  daqmappingiov(ps.getParameter<unsigned long long>("daqmappingiov")),
  record_map(ps.getParameter<string>("record_map")),
  record_mask(ps.getParameter<string>("record_mask")),
  label(ps.getParameter<string>("label")),
  tokenMapping(esConsumes<TotemDAQMapping, TotemReadoutRcd>(edm::ESInputTag("", label))),
  tokenAnalysisMask(esConsumes<TotemAnalysisMask, TotemReadoutRcd>(edm::ESInputTag("", label))) {}


void WriteCTPPSTotemDAQMappingMask::analyze(const edm::Event &, edm::EventSetup const &es) {
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if(auto mappingHandle = es.getHandle(tokenMapping)){
    const auto &mapping = es.getData(tokenMapping);
    printMapping(mapping);
    
    if(poolDbService.isAvailable())
      poolDbService->writeOneIOV(mapping, daqmappingiov, record_map);
    else
      edm::LogError("WriteCTPPSTotemDAQMappingMask mask") << "PoolDBService not availible. Data not written.";
  } else 
    edm::LogError("WriteCTPPSTotemDAQMappingMask mapping") << "No mapping found";
  

  if(auto maskHandle = es.getHandle(tokenAnalysisMask)){
    const auto &analysisMask = es.getData(tokenAnalysisMask);
    printMask(analysisMask);

    if(poolDbService.isAvailable()) 
      poolDbService->writeOneIOV(analysisMask, daqmappingiov, record_mask);
    else
      edm::LogError("WriteCTPPSTotemDAQMappingMask mask") << "PoolDBService not availible. Data not written.";
  } else 
    edm::LogError("WriteCTPPSTotemDAQMappingMask mask") << "No analysis mask found";
}


void WriteCTPPSTotemDAQMappingMask::printMapping(TotemDAQMapping mapping){
  for(const auto &p : mapping.VFATMapping)
    LogInfo("TotemDQMMapping VFAT mapping") << "    " << p.first << " -> " << p.second;
  for(const auto &p : mapping.totemTimingChannelMap)
    LogInfo("WriteCTPPSTotemDAQMappingMask channel mapping") << "    " << p.first << " plane " << p.second.plane << " channel " << p.second.channel;  
}


void WriteCTPPSTotemDAQMappingMask::printMask(TotemAnalysisMask analysisMask){
  for(const auto &p : analysisMask.analysisMask)
    edm::LogInfo("WriteCTPPSTotemDAQMappingMask mask") << "    " << p.first << ": fullMask=" << p.second.fullMask
      << ", number of masked channels " << p.second.maskedChannels.size();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(WriteCTPPSTotemDAQMappingMask);