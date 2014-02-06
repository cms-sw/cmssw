/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CondFormats/DataRecord/interface/HcalAllRcds.h"

//
#include "CondCore/CondDB/interface/Serialization.h"

// required for compiling ( the only available constructor in this class ). Can't be used in persistency without this...
namespace cond {
  template <> HcalCalibrationQIEData* createPayload<HcalCalibrationQIEData>( const std::string& payloadTypeName ){
    if( payloadTypeName == "HcalCalibrationQIEData" ) return new HcalCalibrationQIEData(0);
    throwException(std::string("Type mismatch, target object is type \"")+payloadTypeName+"\"",
		   "createPayload" );
  }

}

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
REGISTER_PLUGIN(HcalLutMetadataRcd,HcalLutMetadata);
REGISTER_PLUGIN(HcalDcsRcd, HcalDcsValues);
REGISTER_PLUGIN(HcalDcsMapRcd,HcalDcsMap);
REGISTER_PLUGIN(HcalCholeskyMatricesRcd,HcalCholeskyMatrices);
REGISTER_PLUGIN(HcalCovarianceMatricesRcd,HcalCovarianceMatrices);
REGISTER_PLUGIN(HcalRecoParamsRcd,HcalRecoParams);
REGISTER_PLUGIN(HcalLongRecoParamsRcd,HcalLongRecoParams);
REGISTER_PLUGIN(HcalMCParamsRcd,HcalMCParams);
REGISTER_PLUGIN(HcalFlagHFDigiTimeParamsRcd,HcalFlagHFDigiTimeParams);
REGISTER_PLUGIN(HcalTimingParamsRcd,HcalTimingParams);
