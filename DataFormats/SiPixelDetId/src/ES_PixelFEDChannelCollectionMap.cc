#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "FWCore/Utilities/interface/typelookup.h"

typedef std::unordered_map<std::string,PixelFEDChannelCollection> PixelFEDChannelCollectionMap;

TYPELOOKUP_DATA_REG(PixelFEDChannelCollectionMap);
