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
 *\brief Prints the DAQ mapping loaded by TotemDAQMappingESSourceXML.
 **/
class WriteCTPPSTotemDAQMappingMask: public edm::one::EDAnalyzer<> {
public:
  WriteCTPPSTotemDAQMappingMask(const edm::ParameterSet &ps);
  ~WriteCTPPSTotemDAQMappingMask() override {}

private:
  void analyze(const edm::Event &e, const edm::EventSetup &es) override;
  
  struct ConfigBlockData {
    cond::Time_t daqmappingiov;
    std::string record_map;
    std::string record_mask;
    std::string label;
    edm::ESGetToken<TotemDAQMapping, TotemReadoutRcd> tokenMapping;
    edm::ESGetToken<TotemAnalysisMask, TotemReadoutRcd> tokenAnalysisMask;
  };

  std::vector<ConfigBlockData> data_to_put;
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

WriteCTPPSTotemDAQMappingMask::WriteCTPPSTotemDAQMappingMask(const edm::ParameterSet &ps){
  data_to_put = {};

  for (const auto &it : ps.getParameter<vector<ParameterSet>>("toWrite")) {
    edm::LogInfo("CONSTRUCToR loop ") ;
    ConfigBlockData b;

    b.daqmappingiov = it.getParameter<unsigned long long>("daqmappingiov");
    b.record_map = it.getParameter<string>("record_map");
    b.record_mask = it.getParameter<string>("record_mask");
    b.label = it.getParameter<string>("label");
    b.tokenMapping = esConsumes<TotemDAQMapping, TotemReadoutRcd>(edm::ESInputTag("", b.label));
    b.tokenAnalysisMask = esConsumes<TotemAnalysisMask, TotemReadoutRcd>(edm::ESInputTag("", b.label));
    
    data_to_put.push_back(b);
  }
}

void WriteCTPPSTotemDAQMappingMask::analyze(const edm::Event &, edm::EventSetup const &es) {
  // print mapping
  // Write DAQ Mapping to sqlite file:

  for (auto itToPut = data_to_put.begin(); itToPut != data_to_put.end(); ++itToPut){
    auto mappingHandle = es.getHandle(itToPut->tokenMapping);
    auto maskHandle = es.getHandle(itToPut->tokenAnalysisMask);
    edm::Service<cond::service::PoolDBOutputService> poolDbService;

    if (mappingHandle){
      const auto mapping = *mappingHandle;

      for (const auto &p : mapping.VFATMapping)
        LogInfo("TotemDQMMapping VFAT mapping") << "    " << p.first << " -> " << p.second;
      for (const auto &p : mapping.totemTimingChannelMap)
        LogInfo("WriteCTPPSTotemDAQMappingMask channel mapping") << "    " << p.first << " plane " << p.second.plane << " channel " << p.second.channel;

      if (poolDbService.isAvailable())
        poolDbService->writeOneIOV(mapping, itToPut->daqmappingiov, itToPut->record_map);
    } else {
      edm::LogError("WriteCTPPSTotemDAQMappingMask mapping") << "No mapping found";
    }

    if (maskHandle) {
      const auto analysisMask = *maskHandle;
      for (const auto &p : analysisMask.analysisMask)
        edm::LogInfo("WriteCTPPSTotemDAQMappingMask mask") << "    " << p.first << ": fullMask=" << p.second.fullMask
                                                  << ", number of masked channels " << p.second.maskedChannels.size();

      if (poolDbService.isAvailable() ) {
        poolDbService->writeOneIOV(analysisMask, itToPut->daqmappingiov, itToPut->record_mask);
      }
    } else {
      edm::LogError("WriteCTPPSTotemDAQMappingMask mask") << "No analysis mask found";
    }

    
  }
}
//TODO: separate addition of mask and map to functions


//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(WriteCTPPSTotemDAQMappingMask);