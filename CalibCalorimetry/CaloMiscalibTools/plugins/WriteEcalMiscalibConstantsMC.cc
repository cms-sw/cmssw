// -*- C++ -*-
//
// Package:    WriteEcalMiscalibConstantsMC
// Class:      WriteEcalMiscalibConstantsMC
//
/**\class WriteEcalMiscalibConstantsMC WriteEcalMiscalibConstantsMC.cc CalibCalorimetry/WriteEcalMiscalibConstantsMC/src/WriteEcalMiscalibConstantsMC.cc

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
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteEcalMiscalibConstantsMC.h"

//
// static data member definitions
//

//
// constructors and destructor
//
WriteEcalMiscalibConstantsMC::WriteEcalMiscalibConstantsMC(const edm::ParameterSet& iConfig)
    : newTagRequest_(iConfig.getParameter<std::string>("NewTagRequest")), intercalibConstsToken_(esConsumes()) {}

WriteEcalMiscalibConstantsMC::~WriteEcalMiscalibConstantsMC() {}

//
// member functions
//

// ------------ method called to for each event  ------------
void WriteEcalMiscalibConstantsMC::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  // Intercalib constants
  const EcalIntercalibConstantsMC* Mcal = &iSetup.getData(intercalibConstsToken_);

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    if (poolDbService->isNewTagRequest(newTagRequest_)) {
      edm::LogVerbatim("WriteEcalMiscalibConstantsMC") << "Creating a new IOV";
      poolDbService->createOneIOV<const EcalIntercalibConstantsMC>(*Mcal, poolDbService->beginOfTime(), newTagRequest_);
      edm::LogVerbatim("WriteEcalMiscalibConstantsMC") << "Done";
    } else {
      edm::LogVerbatim("WriteEcalMiscalibConstantsMC") << "Old IOV";
      poolDbService->appendOneIOV<const EcalIntercalibConstantsMC>(*Mcal, poolDbService->currentTime(), newTagRequest_);
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void WriteEcalMiscalibConstantsMC::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void WriteEcalMiscalibConstantsMC::endJob() { edm::LogVerbatim("WriteEcalMiscalibConstantsMC") << "Here is the end"; }
