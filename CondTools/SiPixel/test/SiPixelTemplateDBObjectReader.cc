#include "CondTools/SiPixel/test/SiPixelTemplateDBObjectReader.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

SiPixelTemplateDBObjectReader::SiPixelTemplateDBObjectReader(const edm::ParameterSet& iConfig)
{
}

SiPixelTemplateDBObjectReader::~SiPixelTemplateDBObjectReader()
{
}

void 
SiPixelTemplateDBObjectReader::beginJob(const edm::EventSetup& setup)
{
	
}

void
SiPixelTemplateDBObjectReader::analyze(const edm::Event& iEvent, const edm::EventSetup& setup)
{
	if(SiPixelTemplateDBObjectWatcher_.check(setup)) {
		
		edm::ESHandle<SiPixelTemplateDBObject> templateH;
		setup.get<SiPixelTemplateDBObjectRcd>().get(templateH);
		
		std::cout << *templateH.product() << std::endl;
	}
}

void 
SiPixelTemplateDBObjectReader::endJob()
{
}
