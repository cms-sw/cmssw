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
  std::string record_;
  std::string label_;

  edm::ESGetToken<TotemDAQMapping, TotemReadoutRcd> tokenMapping_;
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

WriteCTPPSTotemDAQMapping::WriteCTPPSTotemDAQMapping(const edm::ParameterSet &ps)
    : daqmappingiov_(ps.getParameter<unsigned long long>("daqmappingiov")),
      record_(ps.getParameter<string>("record")),
      label_(ps.getParameter<string>("label")),
      tokenMapping_(esConsumes<TotemDAQMapping, TotemReadoutRcd>(edm::ESInputTag("", label_))) {}

void WriteCTPPSTotemDAQMapping::analyze(const edm::Event &, edm::EventSetup const &es) {
  // print mapping
  // Write DAQ Mapping to sqlite file:
  const auto &mapping = es.getData(tokenMapping_);

  printf("* VFAT mapping\n");
  for (const auto &p : mapping.VFATMapping)
    cout << "    " << p.first << " -> " << p.second << endl;
  printf("* Channel mapping\n");
  for (const auto &p : mapping.totemTimingChannelMap)
    cout << "    " << p.first << " plane " << p.second.plane << " channel " << p.second.channel << endl;

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    cond::Time_t firstSinceTime=poolDbService->beginOfTime();
    // poolDbService->writeOneIOV(mapping, daqmappingiov_, record_);
    poolDbService->writeOneIOV(mapping, firstSinceTime, record_);
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(WriteCTPPSTotemDAQMapping);
