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
// $Id: WriteEcalMiscalibConstants.cc,v 1.4 2009/12/17 20:59:29 wmtan Exp $
//
//


// system include files

// user include files




#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

// DB includes
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// user include files
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
//For Checks

//this one
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteEcalMiscalibConstants.h"

//
// static data member definitions
//

//
// constructors and destructor
//
WriteEcalMiscalibConstants::WriteEcalMiscalibConstants(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  newTagRequest_ = iConfig.getParameter< std::string > ("NewTagRequest");


}


WriteEcalMiscalibConstants::~WriteEcalMiscalibConstants()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
WriteEcalMiscalibConstants::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  // Intercalib constants
   edm::ESHandle<EcalIntercalibConstants> pIcal;
   iSetup.get<EcalIntercalibConstantsRcd>().get(pIcal);
   const EcalIntercalibConstants* Mcal = pIcal.product();

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( poolDbService.isAvailable() ){
    if ( poolDbService->isNewTagRequest(newTagRequest_) ){
      std::cout<<" Creating a  new one "<<std::endl;
      poolDbService->createNewIOV<const EcalIntercalibConstants>( Mcal, poolDbService->beginOfTime(), poolDbService->endOfTime(),newTagRequest_);
      std::cout<<"Done" << std::endl;
    }else{
      std::cout<<"Old One "<<std::endl;
       poolDbService->appendSinceTime<const EcalIntercalibConstants>( Mcal, poolDbService->currentTime(),newTagRequest_);
    }
  }  
}


// ------------ method called once each job just before starting event loop  ------------
void 
WriteEcalMiscalibConstants::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 

WriteEcalMiscalibConstants::endJob() {
  std::cout << "Here is the end" << std::endl; 
}

