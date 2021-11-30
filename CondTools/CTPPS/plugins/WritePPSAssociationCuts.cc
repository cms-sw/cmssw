/****************************************************************************
* Author: 
*  Grzegorz Sroka
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

#include "CondFormats/PPSObjects/interface/PPSAssociationCuts.h"
#include "CondFormats/DataRecord/interface/PPSAssociationCutsRcd.h"

#include <memory>

class WritePPSAssociationCuts : public edm::one::EDAnalyzer<> {
public:
  explicit WritePPSAssociationCuts(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  edm::ESGetToken<PPSAssociationCuts, PPSAssociationCutsRcd> esToken_;
};

WritePPSAssociationCuts::WritePPSAssociationCuts(const edm::ParameterSet &iConfig)
    : esToken_(esConsumes<PPSAssociationCuts, PPSAssociationCutsRcd>(
          edm::ESInputTag("", iConfig.getParameter<std::string>("label")))) {}

void WritePPSAssociationCuts::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get the data
  const auto &ppsAssociationCuts = iSetup.getData(esToken_);

  // store the data in a DB object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOneIOV(ppsAssociationCuts, poolDbService->currentTime(), "PPSAssociationCutsRcd");
  } else {
    throw cms::Exception("WritePPSAssociationCuts") << "PoolDBService required.";
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(WritePPSAssociationCuts);
