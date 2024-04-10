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

#include "CondFormats/DataRecord/interface/HcalOOTPileupCorrectionRcd.h"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionColl.h"

#include "CondFormats/DataRecord/interface/HcalOOTPileupCompatibilityRcd.h"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionBuffer.h"

#include "CondFormats/DataRecord/interface/HcalOOTPileupCorrectionMapCollRcd.h"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionMapColl.h"

#include "CondFormats/DataRecord/interface/HcalInterpolatedPulseCollRcd.h"
#include "CondFormats/HcalObjects/interface/HcalInterpolatedPulseColl.h"

#include "CondFormats/DataRecord/interface/HBHENegativeEFilterRcd.h"
#include "CondFormats/HcalObjects/interface/HBHENegativeEFilter.h"

#include "CondFormats/DataRecord/interface/HFPhase1PMTParamsRcd.h"
#include "CondFormats/HcalObjects/interface/HFPhase1PMTParams.h"

//
#include "CondCore/CondDB/interface/Serialization.h"

// required for compiling ( the only available constructor in this class ). Can't be used in persistency without this...
namespace cond {
  template <>
  HcalCalibrationQIEData* createPayload<HcalCalibrationQIEData>(const std::string& payloadTypeName) {
    if (payloadTypeName == "HcalCalibrationQIEData")
      return new HcalCalibrationQIEData(nullptr);
    throwException(std::string("Type mismatch, target object is type \"") + payloadTypeName + "\"", "createPayload");
  }

}  // namespace cond

namespace cond::serialization {
  template <>
  std::unique_ptr<HcalCalibrationQIEData> makeClass<HcalCalibrationQIEData>() {
    return std::make_unique<HcalCalibrationQIEData>(nullptr);
  }
}  // namespace cond::serialization

namespace {
  struct InitHcalElectronicsMap {
    void operator()(HcalElectronicsMap& e) { e.initialize(); }
  };
}  // namespace
namespace {
  struct InitHcalDcsMap {
    void operator()(HcalDcsMap& e) { e.initialize(); }
  };
}  // namespace
namespace {
  struct InitHcalFrontEndMap {
    void operator()(HcalFrontEndMap& e) { e.initialize(); }
  };
}  // namespace
namespace {
  struct InitHcalSiPMCharacteristics {
    void operator()(HcalSiPMCharacteristics& e) { e.initialize(); }
  };
}  // namespace

REGISTER_PLUGIN(HcalPedestalsRcd, HcalPedestals);
REGISTER_PLUGIN(HcalPedestalWidthsRcd, HcalPedestalWidths);
REGISTER_PLUGIN(HcalGainsRcd, HcalGains);
REGISTER_PLUGIN(HcalGainWidthsRcd, HcalGainWidths);
REGISTER_PLUGIN(HcalPFCutsRcd, HcalPFCuts);
REGISTER_PLUGIN_INIT(HcalElectronicsMapRcd, HcalElectronicsMap, InitHcalElectronicsMap);
REGISTER_PLUGIN_INIT(HcalFrontEndMapRcd, HcalFrontEndMap, InitHcalFrontEndMap);
REGISTER_PLUGIN(HcalChannelQualityRcd, HcalChannelQuality);
REGISTER_PLUGIN(HcalQIEDataRcd, HcalQIEData);
REGISTER_PLUGIN(HcalQIETypesRcd, HcalQIETypes);
REGISTER_PLUGIN(HcalCalibrationQIEDataRcd, HcalCalibrationQIEData);
REGISTER_PLUGIN(HcalZSThresholdsRcd, HcalZSThresholds);
REGISTER_PLUGIN(HcalRespCorrsRcd, HcalRespCorrs);
REGISTER_PLUGIN(HcalLUTCorrsRcd, HcalLUTCorrs);
REGISTER_PLUGIN(HcalPFCorrsRcd, HcalPFCorrs);
REGISTER_PLUGIN(HcalTimeCorrsRcd, HcalTimeCorrs);
REGISTER_PLUGIN(HcalL1TriggerObjectsRcd, HcalL1TriggerObjects);
REGISTER_PLUGIN(HcalValidationCorrsRcd, HcalValidationCorrs);
REGISTER_PLUGIN(HcalLutMetadataRcd, HcalLutMetadata);
REGISTER_PLUGIN(HcalDcsRcd, HcalDcsValues);
REGISTER_PLUGIN_INIT(HcalDcsMapRcd, HcalDcsMap, InitHcalDcsMap);
REGISTER_PLUGIN(HcalRecoParamsRcd, HcalRecoParams);
REGISTER_PLUGIN(HcalLongRecoParamsRcd, HcalLongRecoParams);
REGISTER_PLUGIN(HcalZDCLowGainFractionsRcd, HcalZDCLowGainFractions);
REGISTER_PLUGIN(HcalMCParamsRcd, HcalMCParams);
REGISTER_PLUGIN(HcalFlagHFDigiTimeParamsRcd, HcalFlagHFDigiTimeParams);
REGISTER_PLUGIN(HcalTimingParamsRcd, HcalTimingParams);
REGISTER_PLUGIN(HcalOOTPileupCorrectionRcd, OOTPileupCorrectionColl);
REGISTER_PLUGIN(HcalOOTPileupCompatibilityRcd, OOTPileupCorrectionBuffer);
REGISTER_PLUGIN(HcalOOTPileupCorrectionMapCollRcd, OOTPileupCorrectionMapColl);
REGISTER_PLUGIN(HcalInterpolatedPulseCollRcd, HcalInterpolatedPulseColl);
REGISTER_PLUGIN(HBHENegativeEFilterRcd, HBHENegativeEFilter);
REGISTER_PLUGIN(HcalSiPMParametersRcd, HcalSiPMParameters);
REGISTER_PLUGIN_INIT(HcalSiPMCharacteristicsRcd, HcalSiPMCharacteristics, InitHcalSiPMCharacteristics);
REGISTER_PLUGIN(HcalTPParametersRcd, HcalTPParameters);
REGISTER_PLUGIN(HcalTPChannelParametersRcd, HcalTPChannelParameters);
REGISTER_PLUGIN(HFPhase1PMTParamsRcd, HFPhase1PMTParams);  // is HcalItemCollById<HFPhase1PMTData>
