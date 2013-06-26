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
// $Id: WriteEcalMiscalibConstantsMC.cc,v 1.2 2009/12/17 20:59:29 wmtan Exp $
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
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
//For Checks

//this one
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteEcalMiscalibConstantsMC.h"

//
// static data member definitions
//

//
// constructors and destructor
//
WriteEcalMiscalibConstantsMC::WriteEcalMiscalibConstantsMC(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  newTagRequest_ = iConfig.getParameter< std::string > ("NewTagRequest");


}


WriteEcalMiscalibConstantsMC::~WriteEcalMiscalibConstantsMC()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
WriteEcalMiscalibConstantsMC::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  // Intercalib constants
   edm::ESHandle<EcalIntercalibConstantsMC> pIcal;
   iSetup.get<EcalIntercalibConstantsMCRcd>().get(pIcal);
   const EcalIntercalibConstantsMC* Mcal = pIcal.product();

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( poolDbService.isAvailable() ){
    if ( poolDbService->isNewTagRequest(newTagRequest_) ){
      std::cout<<" Creating a  new one "<<std::endl;
      poolDbService->createNewIOV<const EcalIntercalibConstantsMC>( Mcal, poolDbService->beginOfTime(), poolDbService->endOfTime(),newTagRequest_);
      std::cout<<"Done" << std::endl;
    }else{
      std::cout<<"Old One "<<std::endl;
       poolDbService->appendSinceTime<const EcalIntercalibConstantsMC>( Mcal, poolDbService->currentTime(),newTagRequest_);
    }
  }  
}


// ------------ method called once each job just before starting event loop  ------------
void 
WriteEcalMiscalibConstantsMC::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 

WriteEcalMiscalibConstantsMC::endJob() {
  std::cout << "Here is the end" << std::endl; 
}

