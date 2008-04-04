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
  SiPixelCPEParmErrors* pSiPixelCPEParmErrors = new SiPixelCPEParmErrors();
	pSiPixelCPEParmErrors->siPixelCPEParmErrors_By.reserve(300);
	pSiPixelCPEParmErrors->siPixelCPEParmErrors_Bx.reserve(300);
	pSiPixelCPEParmErrors->siPixelCPEParmErrors_Fy.reserve(300);
	pSiPixelCPEParmErrors->siPixelCPEParmErrors_Fx.reserve(300);

	std::ifstream in;
		
	in.open(theFileName.c_str());
	
	SiPixelCPEParmErrors::siPixelCPEParmErrorsEntry Entry;
	in >> Entry.bias >> Entry.pix_height >> Entry.ave_qclu >> Entry.sigma >> Entry.sigma >> Entry.rms;

	while(!in.eof()) {
	  if (Entry.bias == 1) pSiPixelCPEParmErrors->siPixelCPEParmErrors_By.push_back(Entry);
	  else if (Entry.bias == 2) pSiPixelCPEParmErrors->siPixelCPEParmErrors_Bx.push_back(Entry);
	  else if (Entry.bias == 3) pSiPixelCPEParmErrors->siPixelCPEParmErrors_Fy.push_back(Entry);
	  else if (Entry.bias == 4) pSiPixelCPEParmErrors->siPixelCPEParmErrors_Fx.push_back(Entry);
	  
	  in >> Entry.bias >> Entry.pix_height >> Entry.ave_qclu >> Entry.sigma >> Entry.sigma >> Entry.rms;
	}
	
	in.close();

	edm::Service<cond::service::PoolDBOutputService> poolDbService;
	if( poolDbService.isAvailable() )
	  {
	    if ( poolDbService->isNewTagRequest("SiPixelCPEParmErrorsRcd") )
	      poolDbService->createNewIOV<SiPixelCPEParmErrors>( pSiPixelCPEParmErrors, poolDbService->endOfTime(),"SiPixelCPEParmErrorsRcd"  );
	    else
	      poolDbService->appendSinceTime<SiPixelCPEParmErrors>( pSiPixelCPEParmErrors, poolDbService->currentTime(),"SiPixelCPEParmErrorsRcd" );
	  }
	else
	  std::cout << "Pool Service Unavailable" << std::endl;
	
}

