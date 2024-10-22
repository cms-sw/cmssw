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

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelAnalysisMask.h"

#include <cstdint>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Prints the Analysis Mask loaded by TotemDAQMappingESSourceXML.
 **/
class WriteCTPPSPixelAnalysisMask : public edm::one::EDAnalyzer<> {
public:
  WriteCTPPSPixelAnalysisMask(const edm::ParameterSet &ps);
  ~WriteCTPPSPixelAnalysisMask() override {}

private:
  void analyze(const edm::Event &e, const edm::EventSetup &es) override;
  cond::Time_t analysismaskiov_;
  std::string record_;
  std::string label_;

  edm::ESGetToken<CTPPSPixelAnalysisMask, CTPPSPixelAnalysisMaskRcd> tokenAnalysisMask_;
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

WriteCTPPSPixelAnalysisMask::WriteCTPPSPixelAnalysisMask(const edm::ParameterSet &ps)
    : analysismaskiov_(ps.getParameter<unsigned long long>("analysismaskiov")),
      record_(ps.getParameter<string>("record")),
      label_(ps.getParameter<string>("label")),
      tokenAnalysisMask_(esConsumes<CTPPSPixelAnalysisMask, CTPPSPixelAnalysisMaskRcd>(edm::ESInputTag("", label_))) {}

void WriteCTPPSPixelAnalysisMask::analyze(const edm::Event &, edm::EventSetup const &es) {
  /*// print analysisMask
  printf("* mask\n");
  for (const auto &p : analysisMask->analysisMask)
    cout << "    " << p.first
      << ": fullMask=" << p.second.fullMask
      << ", number of masked channels " << p.second.maskedPixels.size() << endl;
  */

  // Write Analysis Mask to sqlite file:

  const auto &analysisMask = es.getData(tokenAnalysisMask_);
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOneIOV(analysisMask, analysismaskiov_, record_);
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(WriteCTPPSPixelAnalysisMask);
