// -*- C++ -*-
//
// Package:    WriteEcalMiscalibConstants
// Class:      WriteEcalMiscalibConstants
//
/**\class WriteEcalMiscalibConstants WriteEcalMiscalibConstants.cc CalibCalorimetry/WriteEcalMiscalibConstants/src/WriteEcalMiscalibConstants.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Stephanie BEAUCERON
//         Created:  Tue May 15 16:23:21 CEST 2007
//
//

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DB includes
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// user include files
//this one
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteEcalMiscalibConstants.h"

//
// static data member definitions
//

//
// constructors and destructor
//
WriteEcalMiscalibConstants::WriteEcalMiscalibConstants(const edm::ParameterSet& iConfig)
    : newTagRequest_(iConfig.getParameter<std::string>("NewTagRequest")), intercalibConstsToken_(esConsumes()) {}

WriteEcalMiscalibConstants::~WriteEcalMiscalibConstants() {}

//
// member functions
//

// ------------ method called to for each event  ------------
void WriteEcalMiscalibConstants::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  // Intercalib constants
  const EcalIntercalibConstants* Mcal = &iSetup.getData(intercalibConstsToken_);

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    if (poolDbService->isNewTagRequest(newTagRequest_)) {
      edm::LogVerbatim("WriteEcalMiscalibConstants") << "Creating a new IOV";
      poolDbService->createOneIOV<const EcalIntercalibConstants>(*Mcal, poolDbService->beginOfTime(), newTagRequest_);
      edm::LogVerbatim("WriteEcalMiscalibConstants") << "Done";
    } else {
      edm::LogVerbatim("WriteEcalMiscalibConstants") << "Old IOV";
      poolDbService->appendOneIOV<const EcalIntercalibConstants>(*Mcal, poolDbService->currentTime(), newTagRequest_);
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void WriteEcalMiscalibConstants::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void WriteEcalMiscalibConstants::endJob() { edm::LogVerbatim("WriteEcalMiscalibConstants") << "Here is the end"; }
