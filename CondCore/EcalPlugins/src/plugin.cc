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

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(EcalPedestalsRcd,EcalPedestals);
REGISTER_PLUGIN(EcalWeightXtalGroupsRcd,EcalWeightXtalGroups);
REGISTER_PLUGIN(EcalTBWeightsRcd,EcalTBWeights);
REGISTER_PLUGIN(EcalGainRatiosRcd,EcalGainRatios);
REGISTER_PLUGIN(EcalIntercalibConstantsRcd,EcalIntercalibConstants);
REGISTER_PLUGIN(EcalADCToGeVConstantRcd,EcalADCToGeVConstant);
