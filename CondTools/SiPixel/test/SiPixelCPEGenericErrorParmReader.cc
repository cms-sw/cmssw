#include "CondTools/SiPixel/test/SiPixelCPEGenericErrorParmReader.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

SiPixelCPEGenericErrorParmReader::SiPixelCPEGenericErrorParmReader(const edm::ParameterSet& iConfig)
{
}

SiPixelCPEGenericErrorParmReader::~SiPixelCPEGenericErrorParmReader()
{
}

void 
SiPixelCPEGenericErrorParmReader::beginJob(const edm::EventSetup& setup)
{
}

void
SiPixelCPEGenericErrorParmReader::analyze(const edm::Event& iEvent, const edm::EventSetup& setup)
{
	if(SiPixelCPEGenericErrorParmWatcher_.check(setup)) {
		
		edm::ESHandle<SiPixelCPEGenericErrorParm> errorsH;
		setup.get<SiPixelCPEGenericErrorParmRcd>().get(errorsH);
	
		std::cout << *errorsH.product();
	}
}

void 
SiPixelCPEGenericErrorParmReader::endJob()
{
}
