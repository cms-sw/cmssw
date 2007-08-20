// Package:    SiPixelErrorEstimation
// Class:      PxCPEdbReader
// 
/**\class PxCPEdbReader PxCPEdbReader.cc CalibTracker/SiPixelErrorEstimation/plugins/PxCPEdbReader.cc

 Description: Retrieves Pixel CPE Parametrization Errors from a database

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "David Fehling"
//         Created:  Fri Aug  17 8:34:48 CDT 2007


// user include files

#include <iostream>
#include <fstream>

#include "CalibTracker/SiPixelErrorEstimation/interface/PxCPEdbReader.h"

#include "CondFormats/SiPixelObjects/interface/PixelCPEParmErrors.h"
#include "CondFormats/DataRecord/interface/PixelCPEParmErrorsRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/EventSetup.h"

PxCPEdbReader::PxCPEdbReader(const edm::ParameterSet& iConfig)
{
}

PxCPEdbReader::~PxCPEdbReader()
{
}

void
PxCPEdbReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

void 
PxCPEdbReader::beginJob(const edm::EventSetup& setup)
{
	edm::ESHandle<PixelCPEParmErrors> PixelCPEParmErrorsHandle;
	setup.get<PixelCPEParmErrorsRcd>().get(PixelCPEParmErrorsHandle);
	
	const std::vector<PixelCPEParmErrors::pixelCPEParmErrorsEntry>& errors = PixelCPEParmErrorsHandle->pixelCPEParmErrors;
	
  PixelCPEParmErrors::pixelCPEParmErrorsEntry Entry;
	
	for (unsigned int i=0;i<errors.size();++i)
	{
		Entry = errors[i];
		std::cout
			<< Entry.part << " "
			<< Entry.size << " "
			<< Entry.alpha << " "
			<< Entry.beta << " "
			<< Entry.sigma << std::endl;
	}
}

void 
PxCPEdbReader::endJob()
{
}
