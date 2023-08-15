/****************************************************************************
*
* Offline analyzer for writing TOTEM DAQ Mapping sqlite file 
*
****************************************************************************/

#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/TotemAnalysisMaskRcd.h"
#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "CondFormats/PPSObjects/interface/TotemAnalysisMask.h"
#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include <cstdint>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Prints and writes to SQLite the DAQ mapping loaded by TotemDAQMappingESSourceXML.
 **/
class WriteCTPPSTotemDAQMappingMask : public edm::one::EDAnalyzer<> {
public:
  WriteCTPPSTotemDAQMappingMask(const edm::ParameterSet &ps);
  ~WriteCTPPSTotemDAQMappingMask() override {}

private:
  cond::Time_t daqMappingIov;
  std::string recordMap;
  std::string recordMask;
  std::string label;
  edm::ESGetToken<TotemDAQMapping, TotemReadoutRcd> tokenMapping;
  edm::ESGetToken<TotemAnalysisMask, TotemReadoutRcd> tokenAnalysisMask;

  void analyze(const edm::Event &e, const edm::EventSetup &es) override;
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

WriteCTPPSTotemDAQMappingMask::WriteCTPPSTotemDAQMappingMask(const edm::ParameterSet &ps)
    : daqMappingIov(ps.getParameter<unsigned long long>("daqMappingIov")),
      recordMap(ps.getParameter<string>("recordMap")),
      recordMask(ps.getParameter<string>("recordMask")),
      label(ps.getParameter<string>("label")),
      tokenMapping(esConsumes<TotemDAQMapping, TotemReadoutRcd>(edm::ESInputTag("", label))),
      tokenAnalysisMask(esConsumes<TotemAnalysisMask, TotemReadoutRcd>(edm::ESInputTag("", label))) {}

void WriteCTPPSTotemDAQMappingMask::analyze(const edm::Event &, edm::EventSetup const &es) {
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (auto mappingHandle = es.getHandle(tokenMapping)) {
    const auto &mapping = es.getData(tokenMapping);
    edm::LogInfo("WriteCTPPSTotemDAQMappingMask mapping");
    mapping.print(std::cout, label);

    if (poolDbService.isAvailable()) {
      poolDbService->writeOneIOV(mapping, daqMappingIov, recordMap);
    } else {
      edm::LogError("WriteCTPPSTotemDAQMappingMask mask")
          << "WriteCTPPSTotemDAQMappingMask: PoolDBService not availible. Data not written.";
    }

  } else {
    edm::LogError("WriteCTPPSTotemDAQMappingMask mapping") << "WriteCTPPSTotemDAQMappingMask: No mapping found";
  }

  if (auto maskHandle = es.getHandle(tokenAnalysisMask)) {
    const auto &analysisMask = es.getData(tokenAnalysisMask);
    edm::LogInfo("WriteCTPPSTotemDAQMappingMask mask") << analysisMask;

    if (poolDbService.isAvailable()) {
      poolDbService->writeOneIOV(analysisMask, daqMappingIov, recordMask);
    } else {
      edm::LogError("WriteCTPPSTotemDAQMappingMask mask")
          << "WriteCTPPSTotemDAQMappingMask: PoolDBService not availible. Data not written.";
    }
  } else {
    edm::LogError("WriteCTPPSTotemDAQMappingMask mask") << "WriteCTPPSTotemDAQMappingMask: No analysis mask found";
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(WriteCTPPSTotemDAQMappingMask);