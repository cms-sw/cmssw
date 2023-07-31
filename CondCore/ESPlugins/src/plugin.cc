/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"

#include "CondFormats/ESObjects/interface/ESWeightStripGroups.h"
#include "CondFormats/DataRecord/interface/ESWeightStripGroupsRcd.h"

#include "CondFormats/ESObjects/interface/ESCondObjectContainer.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/ESAngleCorrectionFactorsRcd.h"

#include "CondFormats/ESObjects/interface/ESADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/ESADCToGeVConstantRcd.h"

#include "CondFormats/ESObjects/interface/ESMIPToGeVConstant.h"
#include "CondFormats/DataRecord/interface/ESMIPToGeVConstantRcd.h"

#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/DataRecord/interface/ESGainRcd.h"

#include "CondFormats/ESObjects/interface/ESThresholds.h"
#include "CondFormats/DataRecord/interface/ESThresholdsRcd.h"

#include "CondFormats/ESObjects/interface/ESTimeSampleWeights.h"
#include "CondFormats/DataRecord/interface/ESTimeSampleWeightsRcd.h"

#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/ESEEIntercalibConstantsRcd.h"

#include "CondFormats/ESObjects/interface/ESMissingEnergyCalibration.h"
#include "CondFormats/DataRecord/interface/ESMissingEnergyCalibrationRcd.h"

#include "CondFormats/ESObjects/interface/ESRecHitRatioCuts.h"
#include "CondFormats/DataRecord/interface/ESRecHitRatioCutsRcd.h"

#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/DataRecord/interface/ESChannelStatusRcd.h"

REGISTER_PLUGIN(ESGainRcd, ESGain);
REGISTER_PLUGIN(ESPedestalsRcd, ESPedestals); //is ESCondObjectContainer<ESPedestal> but needs to match name registered with EventSetup system
REGISTER_PLUGIN(ESTimeSampleWeightsRcd, ESTimeSampleWeights);
REGISTER_PLUGIN(ESIntercalibConstantsRcd, ESFloatCondObjectContainer); //is ESCondObjectContainer<float> but needs to match name registered with EventSetup system
REGISTER_PLUGIN_NO_SERIAL(ESAngleCorrectionFactorsRcd, ESCondObjectContainer<float>);
REGISTER_PLUGIN(ESADCToGeVConstantRcd, ESADCToGeVConstant);
REGISTER_PLUGIN(ESMIPToGeVConstantRcd, ESMIPToGeVConstant);
REGISTER_PLUGIN(ESThresholdsRcd, ESThresholds);
REGISTER_PLUGIN(ESEEIntercalibConstantsRcd, ESEEIntercalibConstants);
REGISTER_PLUGIN(ESMissingEnergyCalibrationRcd, ESMissingEnergyCalibration);
REGISTER_PLUGIN(ESRecHitRatioCutsRcd, ESRecHitRatioCuts);
REGISTER_PLUGIN(ESChannelStatusRcd, ESChannelStatus); //is ESCondObjectContainer<ESChannelStatusCode> but needs to match name registered with EventSetup system
