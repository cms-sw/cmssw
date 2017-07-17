#include "CondTools/SiPixel/test/SiPixelFakeGenErrorDBSourceReader.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <iostream>

SiPixelFakeGenErrorDBSourceReader::SiPixelFakeGenErrorDBSourceReader(const edm::ParameterSet& iConfig)
{
}

SiPixelFakeGenErrorDBSourceReader::~SiPixelFakeGenErrorDBSourceReader()
{
}

void 
SiPixelFakeGenErrorDBSourceReader::beginJob()
{
}

void
SiPixelFakeGenErrorDBSourceReader::analyze(const edm::Event& iEvent, const edm::EventSetup& setup)
{
	if(SiPixelGenErrorDBObjectWatcher_.check(setup)) {
		
	edm::ESHandle<SiPixelGenErrorDBObject> generrorH;
	setup.get<SiPixelGenErrorDBObjectRcd>().get(generrorH);

	std::cout << *generrorH.product() << std::endl;
	}
}

void 
SiPixelFakeGenErrorDBSourceReader::endJob() {
}


