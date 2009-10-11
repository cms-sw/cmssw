
#include <boost/cstdint.hpp>

#include "CondFormats/ESObjects/interface/ESCondObjectContainer.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESWeightStripGroups.h"
#include "CondFormats/ESObjects/interface/ESStripGroupId.h"
#include "CondFormats/ESObjects/interface/ESTBWeights.h"
#include "CondFormats/ESObjects/interface/ESWeightSet.h"
#include "CondFormats/ESObjects/interface/ESADCToGeVConstant.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/ESObjects/interface/ESChannelStatusCode.h"
#include "CondFormats/ESObjects/interface/ESThresholds.h"
#include "CondFormats/ESObjects/interface/ESGain.h"

namespace{
  struct dictionary {

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
 
    ESEEIntercalibConstants eseeintercalib;

    ESChannelStatus channelStatus;

    ESThresholds threshold; 

    ESGain gain;
  };
}
