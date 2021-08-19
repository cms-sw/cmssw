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

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfigRun3v1.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigRun3v1Rcd.h"

#include <memory>

class WritePPSAlignmentConfigRun3v1 : public edm::one::EDAnalyzer<> {
public:
  explicit WritePPSAlignmentConfigRun3v1(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  edm::ESGetToken<PPSAlignmentConfigRun3v1, PPSAlignmentConfigRun3v1Rcd> esToken_;
};

WritePPSAlignmentConfigRun3v1::WritePPSAlignmentConfigRun3v1(const edm::ParameterSet &iConfig)
    : esToken_(esConsumes<PPSAlignmentConfigRun3v1, PPSAlignmentConfigRun3v1Rcd>(
          edm::ESInputTag("", iConfig.getParameter<std::string>("label")))) {}

void WritePPSAlignmentConfigRun3v1::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get the data
  const auto &ppsAlignmentConfigRun3v1 = iSetup.getData(esToken_);

  // store the data in a DB object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOne(&ppsAlignmentConfigRun3v1, poolDbService->currentTime(), "PPSAlignmentConfigRun3v1Rcd");
  } else {
    throw cms::Exception("WritePPSAlignmentConfigRun3v1") << "PoolDBService required.";
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(WritePPSAlignmentConfigRun3v1);
