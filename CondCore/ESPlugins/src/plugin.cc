/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"

#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"

#include "CondFormats/ESObjects/interface/ESWeightStripGroups.h"
#include "CondFormats/DataRecord/interface/ESWeightStripGroupsRcd.h"

#include "CondFormats/ESObjects/interface/ESTBWeights.h"
#include "CondFormats/DataRecord/interface/ESTBWeightsRcd.h"

#include "CondFormats/ESObjects/interface/ESCondObjectContainer.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"

#include "CondFormats/ESObjects/interface/ESADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/ESADCToGeVConstantRcd.h"

#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/DataRecord/interface/ESGainRcd.h"

#include "CondFormats/ESObjects/interface/ESThresholds.h"
#include "CondFormats/DataRecord/interface/ESThresholdsRcd.h"

#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/ESEEIntercalibConstantsRcd.h"

#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/DataRecord/interface/ESChannelStatusRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(ESPedestalsRcd,ESPedestals);
REGISTER_PLUGIN(ESWeightStripGroupsRcd,ESWeightStripGroups);
REGISTER_PLUGIN(ESTBWeightsRcd,ESTBWeights);
REGISTER_PLUGIN(ESIntercalibConstantsRcd,ESFloatCondObjectContainer);
REGISTER_PLUGIN(ESADCToGeVConstantRcd,ESADCToGeVConstant);
REGISTER_PLUGIN(ESThresholdsRcd,ESThresholds);
REGISTER_PLUGIN(ESEEIntercalibConstantsRcd,ESEEIntercalibConstants);
REGISTER_PLUGIN(ESChannelStatusRcd,ESChannelStatus);

