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
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEParmErrors.h"
#include "CondFormats/DataRecord/interface/SiPixelCPEParmErrorsRcd.h"
#include "CondTools/SiPixel/interface/SiPixelDBErrorParametrization.h"

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
	edm::ESHandle<SiPixelCPEParmErrors> errorsH;
	setup.get<SiPixelCPEParmErrorsRcd>().get(errorsH);
	
	const SiPixelCPEParmErrors::DbVector & errors = errorsH->errors();
	const SiPixelCPEParmErrors::DbBinSizeVector & errors_Bin_Size = errorsH->errorsBinSize();
	
	SiPixelCPEParmErrors::DbEntry Entry;
	
	std::cout << "Testing integrity of the database" << std::endl;
	for (unsigned int count=0; count < errors.size(); ++count) {
		Entry = errors[count];
		std::cout
		  << Entry.bias << " "
		  << Entry.pix_height << " "
		  << Entry.ave_Qclus << " "
		  << Entry.sigma << " "
		  << Entry.rms << std::endl;
	}

	std::cout << "\n\nTesting Random Access" << std::endl;
	std::cout << "For Part = 2, Size = 2, alpha = 4 , beta = 3: Sigma = 0.000371" << std::endl;
	unsigned int part = 2;
	unsigned int size = 2;
	unsigned int alpha = 4;
	unsigned int beta = 3;
	unsigned int index = 0;
	
	index = errors_Bin_Size[part-1].partBin_size + errors_Bin_Size[part-1].sizeBin_size * size + errors_Bin_Size[part-1].alphaBin_size * alpha + errors_Bin_Size[part-1].betaBin_size * beta;
		
	Entry = errors[index];
	std::cout
		<< Entry.bias << " "
		<< Entry.pix_height << " "
		<< Entry.ave_Qclus << " "
		<< Entry.sigma << " "
		<< Entry.rms << std::endl;

	SiPixelDBErrorParametrization theErrorGetter;

	theErrorGetter.setDBAccess(setup);

	std::pair<float,float> dbentry;

	dbentry = theErrorGetter.getError(GeomDetEnumerators::PixelBarrel, 3, 3, 1.57, 2.72, true, true);

	std::cout << "\n\nFor Barrel, size x = 3, size y =3, alpha = 1.57, beta = 2.72\n"
						<< "By hand the indices should be for bx and by respectively: 346, 99\n"
						<< "And the errors should be 0.000342 and 0.003106\nThey are: "
						<< dbentry.first << " " << dbentry.second << std::endl;

	
}

void 
PxCPEdbReader::endJob()
{
}
