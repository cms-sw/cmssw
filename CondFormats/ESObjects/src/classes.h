
#include <boost/cstdint.hpp>

#include "CondFormats/ESObjects/interface/ESCondObjectContainer.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESWeightStripGroups.h"
#include "CondFormats/ESObjects/interface/ESStripGroupId.h"
#include "CondFormats/ESObjects/interface/ESTBWeights.h"
#include "CondFormats/ESObjects/interface/ESWeightSet.h"
#include "CondFormats/ESObjects/interface/ESADCToGeVConstant.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/ESObjects/interface/ESChannelStatusCode.h"

namespace{
  struct dictionary {
    uint32_t i32;
 

    ESCondObjectContainer<ESPedestal> ESPedestalsMap;
    ESPedestalsMap::const_iterator ESPedestalsMapIterator;

    ESPedestals pedmap;

 

    ESWeightStripGroups gg;
 
    ESTBWeights tbwgt;
    ESWeightSet wset;
    std::map<  ESStripGroupId,  ESWeightSet > wgmap;
    std::pair< ESStripGroupId,  ESWeightSet > wgmapvalue;
 
    ESADCToGeVConstant adcfactor;
 
    ESIntercalibConstants intercalib;
 
    ESChannelStatus channelStatus;
 
  };
}
