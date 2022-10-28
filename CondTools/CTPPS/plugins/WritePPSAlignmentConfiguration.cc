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

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfiguration.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigurationRcd.h"

#include <memory>

class WritePPSAlignmentConfiguration : public edm::one::EDAnalyzer<> {
public:
  explicit WritePPSAlignmentConfiguration(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  edm::ESGetToken<PPSAlignmentConfiguration, PPSAlignmentConfigurationRcd> esToken_;
};

WritePPSAlignmentConfiguration::WritePPSAlignmentConfiguration(const edm::ParameterSet &iConfig)
    : esToken_(esConsumes<PPSAlignmentConfiguration, PPSAlignmentConfigurationRcd>(
          edm::ESInputTag("", iConfig.getParameter<std::string>("label")))) {}

void WritePPSAlignmentConfiguration::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get the data
  const auto &ppsAlignmentConfiguration = iSetup.getData(esToken_);

  // store the data in a DB object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOneIOV(ppsAlignmentConfiguration, poolDbService->currentTime(), "PPSAlignmentConfigurationRcd");
  } else {
    throw cms::Exception("WritePPSAlignmentConfiguration") << "PoolDBService required.";
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(WritePPSAlignmentConfiguration);
