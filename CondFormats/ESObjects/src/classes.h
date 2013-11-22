#include "CondFormats/ESObjects/src/headers.h"


namespace CondFormats_ESObjects {
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

    ESMIPToGeVConstant mipfactor;
 
    ESIntercalibConstants intercalib;

    ESAngleCorrectionFactors anglecorrection;
 
    ESEEIntercalibConstants eseeintercalib;

    ESMissingEnergyCalibration esmissingecalib;

    ESRecHitRatioCuts esrechitratiocuts;

    ESChannelStatus channelStatus;

    ESThresholds threshold; 

    ESGain gain;

    ESTimeSampleWeights tsweights;
  };
}
