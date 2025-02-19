#include "CondTools/SiPixel/test/SiPixelCPEGenericErrorParmReader.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelCPEGenericDBErrorParametrization.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

SiPixelCPEGenericErrorParmReader::SiPixelCPEGenericErrorParmReader(const edm::ParameterSet& iConfig)
{
}

SiPixelCPEGenericErrorParmReader::~SiPixelCPEGenericErrorParmReader()
{
}

void 
SiPixelCPEGenericErrorParmReader::beginJob()
{
}

void
SiPixelCPEGenericErrorParmReader::analyze(const edm::Event& iEvent, const edm::EventSetup& setup)
{
	if(SiPixelCPEGenericErrorParmWatcher_.check(setup)) {
		
		edm::ESHandle<SiPixelCPEGenericErrorParm> errorsH;
		setup.get<SiPixelCPEGenericErrorParmRcd>().get(errorsH);
	
		std::cout << *errorsH.product();


		//uncomment to test random access
		/*	SiPixelCPEGenericDBErrorParametrization theErrorGetter;
		theErrorGetter.setDBAccess(setup);
		std::pair<float,float> dbentry;
		
		dbentry = theErrorGetter.getError(GeomDetEnumerators::PixelBarrel, 3, 3, 1.57, 2.72, true, true);
		
		std::cout << "\n\n\n---------------------------------------------------------------------\n\n" 
							<< "This test was written for version 1.3. We are using version: " << (*errorsH.product()).version()
							<< "\n\nFor Barrel, size x = 3, size y = 3, alpha = 1.57, beta = 2.72\n"
							<< "By hand the indices should be for bx and by respectively: 346, 99\n"
							<< "And the errors should be 0.000289 and 0.003431\nThey are: "
							<< dbentry.first << " " << dbentry.second << std::endl;
		*/	
	}
}

void 
SiPixelCPEGenericErrorParmReader::endJob()
{
}
