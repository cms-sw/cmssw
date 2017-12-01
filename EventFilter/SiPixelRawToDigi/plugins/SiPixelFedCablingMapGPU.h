// Sushil Dubey, Shashi Dugad, TIFR, December 2017
#ifndef SiPixelFedCablingMapGPUU_H
#define SiPixelFedCablingMapGPU_H

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CablingMapGPU.h"


class SiPixelFedCablingMapGPU {
public:
	SiPixelFedCablingMapGPU(edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap);
	~ SiPixelFedCablingMapGPU() {}
	void process(CablingMap* &cablingMapGPU) ;

private:
	edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap;
	std::vector<unsigned int> fedIds;
	std::unique_ptr<SiPixelFedCablingTree> cabling_;
    const unsigned int MAX_FED = 150;
    const unsigned int MAX_LINK = 48;
    const unsigned int MAX_ROC = 8;
};

#endif