// -*- C++ -*-
//
// Package:    WriteHcalPedestalWidths
// Class:      WriteHcalPedestalWidths
// 
/**\class WriteHcalPedestalWidths WriteHcalPedestalWidths.cc CalibCalorimetry/CaloMiscalibTools/src/WriteHcalPedestalWidths.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Stephanie BEAUCERON
//         Created:  Tue May 15 16:23:21 CEST 2007
// $Id: WriteHcalPedestalWidths.cc,v 1.1 2007/08/02 15:19:10 malgeri Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

// DB includes
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// user include files
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
//For Checks
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//this one
#include "CalibCalorimetry/CaloMiscalibTools/interface/WriteHcalPedestalWidths.h"

//
// static data member definitions
//

//
// constructors and destructor
//
WriteHcalPedestalWidths::WriteHcalPedestalWidths(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  newTagRequest_ = iConfig.getParameter< std::string > ("NewTagRequest");


}


WriteHcalPedestalWidths::~WriteHcalPedestalWidths()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
WriteHcalPedestalWidths::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
WriteHcalPedestalWidths::beginJob(const edm::EventSetup& iSetup)
{
   using namespace edm;
  // Intercalib constants
   edm::ESHandle<HcalPedestalWidths> pIcal;
   iSetup.get<HcalPedestalWidthsRcd>().get(pIcal);

   //   const HcalPedestalWidths* Mcal = pIcal.product();

   HcalPedestalWidths* Mcal = new HcalPedestalWidths(*(pIcal.product()));
   
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( poolDbService.isAvailable() ){
    if ( poolDbService->isNewTagRequest(newTagRequest_) ){
      std::cout<<" Creating a  new one "<<std::endl;
      poolDbService->createNewIOV<const HcalPedestalWidths>( Mcal, poolDbService->endOfTime(),newTagRequest_);
      std::cout<<"Done" << std::endl;
    }else{
      std::cout<<"Old One "<<std::endl;
       poolDbService->appendSinceTime<const HcalPedestalWidths>( Mcal, poolDbService->currentTime(),newTagRequest_);
    }
  }  
}

// ------------ method called once each job just after ending the event loop  ------------
void 

WriteHcalPedestalWidths::endJob() {
  std::cout << "Here is the end" << std::endl; 
}

