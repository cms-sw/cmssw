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
// $Id: WritePedestalsToDB.cc,v 1.1 2007/06/05 15:54:44 boeriu Exp $
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
#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "CondFormats/DataRecord/interface/CSCPedestalsRcd.h"
//For Checks
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//this one
#include "CalibMuon/CSCCalibration/interface/WritePedestalsToDB.h"

//
// static data member definitions
//

//
// constructors and destructor
//
WritePedestalsToDB::WritePedestalsToDB(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  newTagRequest_ = iConfig.getParameter< std::string > ("NewTagRequest");
}


WritePedestalsToDB::~WritePedestalsToDB()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//
// member functions
//

// ------------ method called to for each event  ------------
void WritePedestalsToDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  // Intercalib constants
  edm::ESHandle<CSCPedestals> pedestals;
  iSetup.get<CSCPedestalsRcd>().get(pedestals);
  const CSCPedestals* mypedestals = pedestals.product();
  
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if(poolDbService.isAvailable() ){
    if (poolDbService->isNewTagRequest(newTagRequest_) ){
      std::cout<<" Creating a  new one "<<std::endl;
      poolDbService->createNewIOV<const CSCPedestals>(mypedestals, poolDbService->endOfTime(),newTagRequest_);
      std::cout<<"Done" << std::endl;
    }else{
      std::cout<<"Old One "<<std::endl;
      poolDbService->appendSinceTime<const CSCPedestals>(mypedestals, poolDbService->currentTime(),newTagRequest_);
    }
  }  
}


// ------------ method called once each job just before starting event loop  ------------
void WritePedestalsToDB::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void WritePedestalsToDB::endJob() {
  std::cout << "Here is the end" << std::endl; 
}
