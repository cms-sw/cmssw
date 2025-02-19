/*
 *  plugin for DB interface
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/CastorObjects/interface/CastorPedestals.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidths.h"
#include "CondFormats/CastorObjects/interface/CastorGains.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidths.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "CondFormats/CastorObjects/interface/CastorQIEData.h"
#include "CondFormats/CastorObjects/interface/CastorRecoParams.h"
#include "CondFormats/CastorObjects/interface/CastorSaturationCorrs.h"

#include "CondFormats/DataRecord/interface/CastorPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/CastorQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/CastorRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/CastorSaturationCorrsRcd.h"



REGISTER_PLUGIN(CastorPedestalsRcd,CastorPedestals);
REGISTER_PLUGIN(CastorPedestalWidthsRcd,CastorPedestalWidths);
REGISTER_PLUGIN(CastorGainsRcd,CastorGains);
REGISTER_PLUGIN(CastorGainWidthsRcd,CastorGainWidths);
REGISTER_PLUGIN(CastorElectronicsMapRcd,CastorElectronicsMap);
REGISTER_PLUGIN(CastorChannelQualityRcd,CastorChannelQuality);
REGISTER_PLUGIN(CastorQIEDataRcd,CastorQIEData);
REGISTER_PLUGIN(CastorRecoParamsRcd,CastorRecoParams);
REGISTER_PLUGIN(CastorSaturationCorrsRcd,CastorSaturationCorrs);
