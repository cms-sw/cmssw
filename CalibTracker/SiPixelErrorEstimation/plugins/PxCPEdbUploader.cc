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
  pErrors->reserve();   // Default 1000 elements.  Optimize?  &&&

  //--- Open the file
  std::ifstream in;
  in.open(theFileName.c_str());

	int part;
	float version = 1.3;
	
  SiPixelCPEParmErrors::DbEntry Entry;
  in >> part >> Entry.bias >> Entry.pix_height >> Entry.ave_Qclus >> Entry.sigma >> Entry.rms;

  while(!in.eof()) {
    pErrors->push_back( Entry );

    in >> part            >> Entry.bias  >> Entry.pix_height
			 >> Entry.ave_Qclus >> Entry.sigma >> Entry.rms;
  }
  //--- Finished parsing the file, we're done.
  in.close();

	//--- Specify the current binning sizes to use
	SiPixelCPEParmErrors::DbEntryBinSize ErrorsBinSize;
	//--- Part = 1 By
	ErrorsBinSize.partBin_size  =   0;
	ErrorsBinSize.sizeBin_size  =  40;
	ErrorsBinSize.alphaBin_size =  10;
	ErrorsBinSize.betaBin_size  =   1;
	pErrors->push_back_bin(ErrorsBinSize);
  //--- Part = 2 Bx
	ErrorsBinSize.partBin_size  = 240;
	ErrorsBinSize.alphaBin_size =   1;
	ErrorsBinSize.betaBin_size  =  10;
	pErrors->push_back_bin(ErrorsBinSize);
	//--- Part = 3 Fy
	ErrorsBinSize.partBin_size  = 360;
	ErrorsBinSize.alphaBin_size =  10;
	ErrorsBinSize.betaBin_size  =   1;
	pErrors->push_back_bin(ErrorsBinSize);
	//--- Part = 4 Fx
	ErrorsBinSize.partBin_size  = 400;
	ErrorsBinSize.alphaBin_size =   1;
	ErrorsBinSize.betaBin_size  =  10;
	pErrors->push_back_bin(ErrorsBinSize);

	//--- Specify the Version
	pErrors->set_version(version);


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

