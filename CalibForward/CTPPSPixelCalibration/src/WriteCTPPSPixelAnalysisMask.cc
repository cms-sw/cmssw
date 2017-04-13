/****************************************************************************
*
* Offline analyzer for writing CTPPS Analysis Mask sqlite file 
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
 *\brief Prints the Analysis Mask loaded by TotemDAQMappingESSourceXML.
 **/
class WriteCTPPSPixelAnalysisMask : public edm::one::EDAnalyzer<>
{
  public:
    WriteCTPPSPixelAnalysisMask(const edm::ParameterSet &ps);
    ~WriteCTPPSPixelAnalysisMask() {}

  private:
    virtual void analyze(const edm::Event &e, const edm::EventSetup &es) override;
    cond::Time_t analysismaskiov_;
    std::string record_;
    std::string label_;
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

WriteCTPPSPixelAnalysisMask::WriteCTPPSPixelAnalysisMask(const edm::ParameterSet &ps) :
analysismaskiov_(ps.getParameter<unsigned long long>("analysismaskiov")),
record_(ps.getParameter<string>("record")),
label_(ps.getParameter<string>("label"))
{}


void WriteCTPPSPixelAnalysisMask::analyze(const edm::Event&, edm::EventSetup const& es)
{

  // get analysis mask to mask channels
  ESHandle<CTPPSPixelAnalysisMask> analysisMask;
  es.get<CTPPSPixelAnalysisMaskRcd>().get(label_, analysisMask);

  /*// print analysisMask
  printf("* mask\n");
  for (const auto &p : analysisMask->analysisMask)
    cout << "    " << p.first
      << ": fullMask=" << p.second.fullMask
      << ", number of masked channels " << p.second.maskedPixels.size() << endl;
  */

  // Write Analysis Mask to sqlite file:
  const CTPPSPixelAnalysisMask* pCTPPSPixelAnalysisMask = analysisMask.product(); // Analysis Mask
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( poolDbService.isAvailable() ){
    poolDbService->writeOne( pCTPPSPixelAnalysisMask, analysismaskiov_, /*m_record*/ record_  );
  }


}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(WriteCTPPSPixelAnalysisMask);
