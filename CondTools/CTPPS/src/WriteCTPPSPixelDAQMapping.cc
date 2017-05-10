/****************************************************************************
*
* Offline analyzer for writing CTPPS DAQ Mapping sqlite file 
* H. Malbouisson
* based on TOTEM code from  Jan Kašpar (jan.kaspar@gmail.com)
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
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelAnalysisMask.h"

#include <stdint.h>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Prints the DAQ mapping loaded by TotemDAQMappingESSourceXML.
 **/
class WriteCTPPSPixelDAQMapping : public edm::one::EDAnalyzer<>
{
  public:
    WriteCTPPSPixelDAQMapping(const edm::ParameterSet &ps);
    ~WriteCTPPSPixelDAQMapping() {}

  private:
    virtual void analyze(const edm::Event &e, const edm::EventSetup &es) override;
    cond::Time_t daqmappingiov_;
    std::string record_;
    std::string label_;
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

WriteCTPPSPixelDAQMapping::WriteCTPPSPixelDAQMapping(const edm::ParameterSet &ps) :
daqmappingiov_(ps.getParameter<unsigned long long>("daqmappingiov")),
record_(ps.getParameter<string>("record")),
label_(ps.getParameter<string>("label"))
{}


void WriteCTPPSPixelDAQMapping::analyze(const edm::Event&, edm::EventSetup const& es)
{

  // get DAQ mapping
  edm::ESHandle<CTPPSPixelDAQMapping> mapping;
  es.get<CTPPSPixelDAQMappingRcd>().get(label_, mapping);

  // print mapping
  /*printf("* DAQ mapping\n");
  for (const auto &p : mapping->ROCMapping)
    cout << "    " << p.first << " -> " << p.second << endl;
  */

  // Write DAQ Mapping to sqlite file:
  const CTPPSPixelDAQMapping* pCTPPSPixelDAQMapping = mapping.product(); // DAQ Mapping
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( poolDbService.isAvailable() ){
    poolDbService->writeOne( pCTPPSPixelDAQMapping, daqmappingiov_, /*m_record*/ record_  );
  }


}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(WriteCTPPSPixelDAQMapping);
