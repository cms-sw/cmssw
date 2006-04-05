#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

using namespace std;
SiPixelFedCablingMap::SiPixelFedCablingMap() { }

SiPixelFedCablingMap::~SiPixelFedCablingMap() { }

void SiPixelFedCablingMap::addFed(PixelFEDCabling * f)
{
  theFedCablings.push_back(f);
}

//EVENTSETUP_DATA_REG(SiPixelFedCablingMap);
