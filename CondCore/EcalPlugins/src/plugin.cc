/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

// #include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(EcalPedestalsRcd,EcalPedestals);
REGISTER_PLUGIN(EcalWeightXtalGroupsRcd,EcalWeightXtalGroups);
REGISTER_PLUGIN(EcalTBWeightsRcd,EcalTBWeights);
REGISTER_PLUGIN(EcalGainRatiosRcd,EcalGainRatios);
REGISTER_PLUGIN(EcalIntercalibConstantsRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalADCToGeVConstantRcd,EcalADCToGeVConstant);
REGISTER_PLUGIN(EcalLaserAlphasRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalLaserAPDPNRatiosRcd,EcalLaserAPDPNRatios);
REGISTER_PLUGIN(EcalLaserAPDPNRatiosRefRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalChannelStatusRcd,EcalChannelStatus);
