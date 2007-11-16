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
	edm::ESHandle<SiPixelCPEParmErrors> SiPixelCPEParmErrorsHandle;
	setup.get<SiPixelCPEParmErrorsRcd>().get(SiPixelCPEParmErrorsHandle);
	
	const std::vector<SiPixelCPEParmErrors::siPixelCPEParmErrorsEntry>& errors_By = SiPixelCPEParmErrorsHandle->siPixelCPEParmErrors_By;
	const std::vector<SiPixelCPEParmErrors::siPixelCPEParmErrorsEntry>& errors_Bx = SiPixelCPEParmErrorsHandle->siPixelCPEParmErrors_Bx;
	const std::vector<SiPixelCPEParmErrors::siPixelCPEParmErrorsEntry>& errors_Fy = SiPixelCPEParmErrorsHandle->siPixelCPEParmErrors_Fy;
	const std::vector<SiPixelCPEParmErrors::siPixelCPEParmErrorsEntry>& errors_Fx = SiPixelCPEParmErrorsHandle->siPixelCPEParmErrors_Fx;
	
	SiPixelCPEParmErrors::siPixelCPEParmErrorsEntry Entry;
	
	//Part = 1 By
	for (unsigned int size=0;size<6;++size) {
	  for (unsigned int alpha=0;alpha<4;++alpha) {
	    for (unsigned int beta=0;beta<10;++beta) {
	      unsigned int index = 40*size + 10*alpha + beta;
	      if (index > errors_By.size()-1) break;
	      else {
		Entry = errors_By[index];
		std::cout
		  << Entry.bias << " "
		  << Entry.pix_height << " "
		  << Entry.ave_qclu << " "
		  << Entry.sigma << " "
		  << Entry.rms << std::endl;
	      }
	    }
	  }
	}
	//Part = 2 Bx
	for (unsigned int size=0;size<3;++size) {
	  for (unsigned int beta=0;beta<4;++beta) {
	    for (unsigned int alpha=0;alpha<10;++alpha) {
	      unsigned int index = 40*size + 10*beta + alpha;
	      if (index > errors_Bx.size()-1) break;
	      else {
		Entry = errors_Bx[index];
		std::cout
		  << Entry.bias << " "
		  << Entry.pix_height << " "
		  << Entry.ave_qclu << " "
		  << Entry.sigma << " "
		  << Entry.rms << std::endl;
	      }
	    }
	  }
	}
	//Part = 3 Fy
	for (unsigned int size=0;size<1;++size) {
	  for (unsigned int alpha=0;alpha<4;++alpha) {
	    for (unsigned int beta=0;beta<10;++beta) {
	      unsigned int index = 40*size + 10*alpha + beta;
	      if (index > errors_Fy.size()-1) break;
	      else {
		Entry = errors_Fy[index];
		std::cout
		  << Entry.bias << " "
		  << Entry.pix_height << " "
		  << Entry.ave_qclu << " "
		  << Entry.sigma << " "
		  << Entry.rms << std::endl;
	      }
	    }
	  }
	}
	//Part = 4 Fx
	for (unsigned int size=0;size<1;++size) {
	  for (unsigned int beta=0;beta<4;++beta) {
	    for (unsigned int alpha=0;alpha<10;++alpha) {
	      unsigned int index = 40*size + 10*beta + alpha;
	      if (index > errors_Fx.size()-1) break;
	      else {
		Entry = errors_Fx[index];
		std::cout
		  << Entry.bias << " "
		  << Entry.pix_height << " "
		  << Entry.ave_qclu << " "
		  << Entry.sigma << " "
		  << Entry.rms << std::endl;
	      }
	    }
	  }
	}
	
}

void 
PxCPEdbReader::endJob()
{
}
