#ifndef RecoLuminosity_LumiProducer_testSiteService_h
#define RecoLuminosity_LumiProducer_testSiteService_h
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
//#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>

class testSiteService : public edm::one::EDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  explicit testSiteService(edm::ParameterSet const&);

private:
  void beginJob() override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override {}
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;
  void endJob() override;
};  //end class
// -----------------------------------------------------------------
testSiteService::testSiteService(edm::ParameterSet const& iConfig) {}
// -----------------------------------------------------------------
void testSiteService::analyze(edm::Event const& e, edm::EventSetup const&) {}
// -----------------------------------------------------------------
void testSiteService::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) {
  std::cout << "testSiteService::endLuminosityBlock" << std::endl;
  std::cout << "I'm in run " << lumiBlock.run() << " lumi block " << lumiBlock.id().luminosityBlock() << std::endl;
  edm::Service<edm::SiteLocalConfig> localconfservice;
  if (!localconfservice.isAvailable()) {
    throw cms::Exception("edm::SiteLocalConfigService is not available");
  }
  std::string a("frontier://cmsfrontier.cern.ch:8000/CMS_LUMI_DEV_OFFLINE");
  std::cout << "a: " << localconfservice->lookupCalibConnect(a) << std::endl;
  std::string b("frontier://(serverurl=cmsfrontier.cern.ch:8000/LumiPrep)/CMS_LUMI_DEV_OFFLINE");
  std::cout << "b: " << localconfservice->lookupCalibConnect(b) << std::endl;
  ;
  std::string cc("frontier://LumiPrep/CMS_LUMI_DEV_OFFLINE");
  std::cout << "cc: " << localconfservice->lookupCalibConnect(cc) << std::endl;
  std::string dd("frontier://LumiPrep(retrieve-ziplevel=0)/CMS_LUMI_DEV_OFFLINE");
  std::cout << "dd: " << localconfservice->lookupCalibConnect(dd) << std::endl;
}
// -----------------------------------------------------------------
void testSiteService::beginJob() { std::cout << "testEvtLoop::beginJob" << std::endl; }
// -----------------------------------------------------------------
void testSiteService::endJob() {}
DEFINE_FWK_MODULE(testSiteService);
#endif
