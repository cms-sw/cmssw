// -*- C++ -*-
//
// Package:    RecoEcal/EcalSCDynamicDPhiParametersMaker
// Class:      EcalSCDynamicDPhiParametersMaker
//
/**\class EcalSCDynamicDPhiParametersMaker EcalSCDynamicDPhiParametersMaker.cc

 Description: Writes EcalSCDynamicDPhiParameters to the DB

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Reis
//         Created:  Thu, 05 Nov 2020 12:32:03 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/EcalObjects/interface/EcalSCDynamicDPhiParameters.h"
#include "CondFormats/DataRecord/interface/EcalSCDynamicDPhiParametersRcd.h"

class EcalSCDynamicDPhiParametersMaker : public edm::one::EDAnalyzer<> {
public:
  explicit EcalSCDynamicDPhiParametersMaker(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::ESGetToken<EcalSCDynamicDPhiParameters, EcalSCDynamicDPhiParametersRcd> parametersToken_;
};

EcalSCDynamicDPhiParametersMaker::EcalSCDynamicDPhiParametersMaker(const edm::ParameterSet& iConfig)
    : parametersToken_(esConsumes<EcalSCDynamicDPhiParameters, EcalSCDynamicDPhiParametersRcd>()) {}

void EcalSCDynamicDPhiParametersMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<EcalSCDynamicDPhiParameters> esParamsHandle_ = iSetup.getHandle(parametersToken_);

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOne(esParamsHandle_.product(), poolDbService->currentTime(), "EcalSCDynamicDPhiParametersRcd");
  } else {
    throw cms::Exception("PoolDBService") << "No PoolDBService available.";
  }
}

void EcalSCDynamicDPhiParametersMaker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(EcalSCDynamicDPhiParametersMaker);
