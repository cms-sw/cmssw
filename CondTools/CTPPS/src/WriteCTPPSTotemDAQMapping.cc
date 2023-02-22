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
#include <cstdint>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Prints the DAQ mapping loaded by TotemDAQMappingESSourceXML.
 **/
class WriteCTPPSTotemDAQMapping : public edm::one::EDAnalyzer<> {
public:
  WriteCTPPSTotemDAQMapping(const edm::ParameterSet &ps);
  ~WriteCTPPSTotemDAQMapping() override {}

private:
  void analyze(const edm::Event &e, const edm::EventSetup &es) override;
  cond::Time_t daqmappingiov_;
  std::string record_map;
  std::string record_mask;
  std::string label_;

  edm::ESGetToken<TotemDAQMapping, TotemReadoutRcd> tokenMapping_;
  edm::ESGetToken<TotemAnalysisMask, TotemReadoutRcd> tokenAnalysisMask_;
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

WriteCTPPSTotemDAQMapping::WriteCTPPSTotemDAQMapping(const edm::ParameterSet &ps)
    : daqmappingiov_(ps.getParameter<unsigned long long>("daqmappingiov")),
      record_map(ps.getParameter<string>("record_map")),
      record_mask(ps.getParameter<string>("record_mask")),
      label_(ps.getParameter<string>("label")),
      tokenMapping_(esConsumes<TotemDAQMapping, TotemReadoutRcd>(edm::ESInputTag("", label_))),
      tokenAnalysisMask_(esConsumes<TotemAnalysisMask, TotemReadoutRcd>(edm::ESInputTag("", label_))) {}

void WriteCTPPSTotemDAQMapping::analyze(const edm::Event &, edm::EventSetup const &es) {
  // print mapping
  // Write DAQ Mapping to sqlite file:
  const auto &mapping = es.getData(tokenMapping_);
  const auto &analysisMask = es.getData(tokenAnalysisMask_);

  printf("* VFAT mapping\n");
  for (const auto &p : mapping.VFATMapping)
    cout << "    " << p.first << " -> " << p.second << endl;
  printf("* Channel mapping\n");
  for (const auto &p : mapping.totemTimingChannelMap)
    cout << "    " << p.first << " plane " << p.second.plane << " channel " << p.second.channel << endl;

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOneIOV(mapping, daqmappingiov_, record_map);
    poolDbService->writeOneIOV(analysisMask, daqmappingiov_, record_mask);
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(WriteCTPPSTotemDAQMapping);
