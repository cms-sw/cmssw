// -*- C++ -*-
//
// Package:    RecoEcal/EcalMustacheSCParametersMaker
// Class:      EcalMustacheSCParametersMaker
//
/**\class EcalMustacheSCParametersMaker EcalMustacheSCParametersMaker.cc

 Description: Writes EcalMustacheSCParameters to the DB

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Reis
//         Created:  Thu, 05 Nov 2020 09:18:19 GMT
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
#include "CondFormats/EcalObjects/interface/EcalMustacheSCParameters.h"
#include "CondFormats/DataRecord/interface/EcalMustacheSCParametersRcd.h"

class EcalMustacheSCParametersMaker : public edm::one::EDAnalyzer<> {
public:
  explicit EcalMustacheSCParametersMaker(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::ESGetToken<EcalMustacheSCParameters, EcalMustacheSCParametersRcd> parametersToken_;
};

EcalMustacheSCParametersMaker::EcalMustacheSCParametersMaker(const edm::ParameterSet& iConfig)
    : parametersToken_(esConsumes<EcalMustacheSCParameters, EcalMustacheSCParametersRcd>()) {}

void EcalMustacheSCParametersMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<EcalMustacheSCParameters> esParamsHandle_ = iSetup.getHandle(parametersToken_);

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOne(esParamsHandle_.product(), poolDbService->currentTime(), "EcalMustacheSCParametersRcd");
  } else {
    throw cms::Exception("PoolDBService") << "No PoolDBService available.";
  }
}

void EcalMustacheSCParametersMaker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(EcalMustacheSCParametersMaker);
