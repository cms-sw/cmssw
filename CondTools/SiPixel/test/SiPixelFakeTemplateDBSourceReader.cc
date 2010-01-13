#include "CondTools/SiPixel/test/SiPixelFakeTemplateDBSourceReader.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

SiPixelFakeTemplateDBSourceReader::SiPixelFakeTemplateDBSourceReader(const edm::ParameterSet& iConfig)
{
}

SiPixelFakeTemplateDBSourceReader::~SiPixelFakeTemplateDBSourceReader()
{
}

void 
SiPixelFakeTemplateDBSourceReader::beginJob()
{
}

void
SiPixelFakeTemplateDBSourceReader::analyze(const edm::Event& iEvent, const edm::EventSetup& setup)
{
	if(SiPixelTemplateDBObjectWatcher_.check(setup)) {
		
	edm::ESHandle<SiPixelTemplateDBObject> templateH;
	setup.get<SiPixelTemplateDBObjectRcd>().get(templateH);

	std::cout << *templateH.product() << std::endl;
	}
}

void 
SiPixelFakeTemplateDBSourceReader::endJob() {
}


