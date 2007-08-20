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

#include "CondFormats/SiPixelObjects/interface/PixelCPEParmErrors.h"
#include "CondFormats/DataRecord/interface/PixelCPEParmErrorsRcd.h"

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
	PixelCPEParmErrors* pPixelCPEParmErrors = new PixelCPEParmErrors();
	pPixelCPEParmErrors->pixelCPEParmErrors.reserve(2000);
	std::ifstream in;
		
	in.open(theFileName.c_str());
	
	PixelCPEParmErrors::pixelCPEParmErrorsEntry Entry;
	in >> Entry.part >> Entry.size >> Entry.alpha >> Entry.beta >> Entry.sigma;
	while(!in.eof()) {
		pPixelCPEParmErrors->pixelCPEParmErrors.push_back(Entry);
		in >> Entry.part >> Entry.size >> Entry.alpha >> Entry.beta >> Entry.sigma;
	}
	
	in.close();
	
	edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( poolDbService.isAvailable() )
	{
    if ( poolDbService->isNewTagRequest("PixelCPEParmErrorsRcd") )
			poolDbService->createNewIOV<PixelCPEParmErrors>( pPixelCPEParmErrors, poolDbService->endOfTime(),"PixelCPEParmErrorsRcd"  );
    else
			poolDbService->appendSinceTime<PixelCPEParmErrors>( pPixelCPEParmErrors, poolDbService->currentTime(),"PixelCPEParmErrorsRcd" );
	}
	else
		std::cout << "Pool Service Unavailable" << std::endl;
	
}

