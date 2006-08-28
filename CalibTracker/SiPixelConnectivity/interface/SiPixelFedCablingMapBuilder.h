#ifndef SiPixelFedCablingMapBuilder_H
#define SiPixelFedCablingMapBuilder_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <string>

class PixelModuleName;

class SiPixelFedCablingMapBuilder{
public:
  SiPixelFedCablingMapBuilder(const std::string & associatorName);
  SiPixelFedCablingMap * produce(const edm::EventSetup& setup);  
private:
  struct FedSpec {
      int fedId;                             // fed ID
      std::vector<PixelModuleName* > names;  // names of modules
      std::vector<uint32_t> rawids;          // modules corresponding to names
  };
  std::string theAssociatorName;
};

#endif
