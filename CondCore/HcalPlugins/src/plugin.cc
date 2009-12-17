/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CondFormats/DataRecord/interface/HcalAllRcds.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(HcalPedestalsRcd,HcalPedestals);
REGISTER_PLUGIN(HcalPedestalWidthsRcd,HcalPedestalWidths);
REGISTER_PLUGIN(HcalGainsRcd,HcalGains);
REGISTER_PLUGIN(HcalGainWidthsRcd,HcalGainWidths);
REGISTER_PLUGIN(HcalElectronicsMapRcd,HcalElectronicsMap);
REGISTER_PLUGIN(HcalChannelQualityRcd,HcalChannelQuality);
REGISTER_PLUGIN(HcalQIEDataRcd,HcalQIEData);
REGISTER_PLUGIN(HcalCalibrationQIEDataRcd,HcalCalibrationQIEData);
REGISTER_PLUGIN(HcalZSThresholdsRcd,HcalZSThresholds);
REGISTER_PLUGIN(HcalRespCorrsRcd,HcalRespCorrs);
REGISTER_PLUGIN(HcalLUTCorrsRcd,HcalLUTCorrs);
REGISTER_PLUGIN(HcalPFCorrsRcd,HcalPFCorrs);
REGISTER_PLUGIN(HcalTimeCorrsRcd,HcalTimeCorrs);
REGISTER_PLUGIN(HcalL1TriggerObjectsRcd,HcalL1TriggerObjects);
REGISTER_PLUGIN(HcalValidationCorrsRcd,HcalValidationCorrs);
