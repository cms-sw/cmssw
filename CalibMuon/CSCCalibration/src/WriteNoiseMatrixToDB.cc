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
// $Id: WriteNoiseMatrixToDB.cc,v 1.1 2007/06/05 15:54:44 boeriu Exp $
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
#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"
//For Checks
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//this one
#include "CalibMuon/CSCCalibration/interface/WriteNoiseMatrixToDB.h"

//
// static data member definitions
//

//
// constructors and destructor
//
WriteNoiseMatrixToDB::WriteNoiseMatrixToDB(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  newTagRequest_ = iConfig.getParameter< std::string > ("NewTagRequest");
}


WriteNoiseMatrixToDB::~WriteNoiseMatrixToDB()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//
// member functions
//

// ------------ method called to for each event  ------------
void WriteNoiseMatrixToDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  // Intercalib constants
  edm::ESHandle<CSCNoiseMatrix> matrix;
  iSetup.get<CSCNoiseMatrixRcd>().get(matrix);
  const CSCNoiseMatrix* mymatrix = matrix.product();
  
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if(poolDbService.isAvailable() ){
    if (poolDbService->isNewTagRequest(newTagRequest_) ){
      std::cout<<" Creating a  new one "<<std::endl;
      poolDbService->createNewIOV<const CSCNoiseMatrix>(mymatrix, poolDbService->endOfTime(),newTagRequest_);
      std::cout<<"Done" << std::endl;
    }else{
      std::cout<<"Old One "<<std::endl;
      poolDbService->appendSinceTime<const CSCNoiseMatrix>(mymatrix, poolDbService->currentTime(),newTagRequest_);
    }
  }  
}


// ------------ method called once each job just before starting event loop  ------------
void WriteNoiseMatrixToDB::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void WriteNoiseMatrixToDB::endJob() {
  std::cout << "Here is the end" << std::endl; 
}
