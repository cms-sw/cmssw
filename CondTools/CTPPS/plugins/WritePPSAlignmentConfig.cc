/****************************************************************************
* Author: 
*  Mateusz Kocot (mateuszkocot99@gmail.com)
****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfig.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigRcd.h"

#include <memory>

class WritePPSAlignmentConfig : public edm::one::EDAnalyzer<> {
public:
  explicit WritePPSAlignmentConfig(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  edm::ESGetToken<PPSAlignmentConfig, PPSAlignmentConfigRcd> esToken_;
};

WritePPSAlignmentConfig::WritePPSAlignmentConfig(const edm::ParameterSet &iConfig)
    : esToken_(esConsumes<PPSAlignmentConfig, PPSAlignmentConfigRcd>(
          edm::ESInputTag("", iConfig.getParameter<std::string>("label")))) {}

void WritePPSAlignmentConfig::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get the data
  const auto &ppsAlignmentConfig = iSetup.getData(esToken_);

  // store the data in a DB object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOne(&ppsAlignmentConfig, poolDbService->currentTime(), "PPSAlignmentConfigRcd");
  } else {
    throw cms::Exception("WritePPSAlignmentConfig") << "PoolDBService required.";
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(WritePPSAlignmentConfig);
