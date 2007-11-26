// Package:    SiPixelErrorEstimation
// Class:      PxCPEdbUploader
// 
/**\class PxCPEdbUploader PxCPEdbUploader.cc CalibTracker/SiPixelErrorEstimation/plugins/PxCPEUploader.cc

 Description: Uploads Pixel CPE Parametrization Errors to a database

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "David Fehling"
//         Created:  Fri Aug  17 8:34:48 CDT 2007


// user include files
#include <iostream>
#include <fstream>

#include "CalibTracker/SiPixelErrorEstimation/interface/PxCPEdbUploader.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelCPEParmErrors.h"
#include "CondFormats/DataRecord/interface/SiPixelCPEParmErrorsRcd.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


PxCPEdbUploader::PxCPEdbUploader(const edm::ParameterSet& iConfig):
	theFileName( iConfig.getParameter<std::string>("fileName") )
{
}


PxCPEdbUploader::~PxCPEdbUploader()
{
}

void
PxCPEdbUploader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

void 
PxCPEdbUploader::beginJob(const edm::EventSetup&)
{
}

void 
PxCPEdbUploader::endJob()
{
  //--- Make the POOL-ORA thingy to store the vector of error structs (DbEntry)
  SiPixelCPEParmErrors* pErrors = new SiPixelCPEParmErrors();
  pErrors->reserve();   // Default 300 elements.  Optimize?  &&&

  //--- Open the file
  std::ifstream in;
  in.open(theFileName.c_str());
	
  SiPixelCPEParmErrors::DbEntry Entry;
  in >> Entry.bias >> Entry.pix_height >> Entry.ave_qclu >> Entry.sigma >> Entry.sigma >> Entry.rms;

  while(!in.eof()) {
    //--- [Petar] I don't understand why Entry.bias carries this info?
    pErrors->push_back( (int)Entry.bias, Entry );
    //
    in >> Entry.bias  >> Entry.pix_height >> Entry.ave_qclu 
       >> Entry.sigma >> Entry.sigma      >> Entry.rms;
  }
  //--- Finished parsing the file, we're done.
  in.close();
  

  //--- Create a new IOV
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if( poolDbService.isAvailable() ) {
    if ( poolDbService->isNewTagRequest("SiPixelCPEParmErrorsRcd") )
      poolDbService->
	createNewIOV<SiPixelCPEParmErrors>( pErrors, 
					    poolDbService->endOfTime(),
					    "SiPixelCPEParmErrorsRcd"  );
    else
      poolDbService->
	appendSinceTime<SiPixelCPEParmErrors>( pErrors, 
					       poolDbService->currentTime(),
					       "SiPixelCPEParmErrorsRcd" );
  }
  else {
    std::cout << "Pool Service Unavailable" << std::endl;
    // &&& throw an exception???
  }
}

