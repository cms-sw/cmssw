#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondTools/SiPixel/test/SiPixelFakeCPEGenericErrorParmSourceReader.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"
#include "CondFormats/DataRecord/interface/SiPixelCPEGenericErrorParmRcd.h"

SiPixelFakeCPEGenericErrorParmSourceReader::SiPixelFakeCPEGenericErrorParmSourceReader(const edm::ParameterSet& iConfig)
{
}


SiPixelFakeCPEGenericErrorParmSourceReader::~SiPixelFakeCPEGenericErrorParmSourceReader()
{
}

void 
SiPixelFakeCPEGenericErrorParmSourceReader::beginJob()
{
}

void
SiPixelFakeCPEGenericErrorParmSourceReader::analyze(const edm::Event& iEvent, const edm::EventSetup& setup)
{
	if(SiPixelCPEGenericErrorParmWatcher_.check(setup)) {
		
		edm::ESHandle<SiPixelCPEGenericErrorParm> errorsH;
		setup.get<SiPixelCPEGenericErrorParmRcd>().get(errorsH);

		std::cout << *errorsH << std::endl;
	}
}

void 
SiPixelFakeCPEGenericErrorParmSourceReader::endJob()
{
}

