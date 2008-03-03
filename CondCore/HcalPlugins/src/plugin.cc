/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalCalibrationQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalZSThresholds.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"

#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalCalibrationQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalZSThresholdsRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"

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
