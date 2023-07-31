#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/DataRecord/interface/SiPixelCPEGenericErrorParmRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelCalibConfigurationRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelDynamicInefficiencyRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationOfflineRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelPerformanceSummaryRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelStatusScenarioProbabilityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelStatusScenariosRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelVCalRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDynamicInefficiency.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFEDChannelContainer.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQualityProbabilities.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"

#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"

#include "CondFormats/DataRecord/interface/SiPixel2DTemplateDBObjectRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"

#include "CondFormats/DataRecord/interface/PixelDCSRcds.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"

#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTSimRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationOfflineSimRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelVCalSimRcd.h"

namespace {
  struct InitRocs {
    void operator()(SiPixelFedCablingMap& m) { m.initializeRocs(); }
  };
  template <typename G>
  struct InitGains {
    void operator()(G& g) { g.initialize(); }
  };
}  // namespace

REGISTER_PLUGIN_INIT(SiPixelFedCablingMapRcd, SiPixelFedCablingMap, InitRocs);
REGISTER_PLUGIN_INIT(SiPixelGainCalibrationRcd, SiPixelGainCalibration, InitGains<SiPixelGainCalibration>);
REGISTER_PLUGIN_INIT(SiPixelGainCalibrationForHLTRcd,
                     SiPixelGainCalibrationForHLT,
                     InitGains<SiPixelGainCalibrationForHLT>);
REGISTER_PLUGIN_INIT(SiPixelGainCalibrationOfflineRcd,
                     SiPixelGainCalibrationOffline,
                     InitGains<SiPixelGainCalibrationOffline>);
REGISTER_PLUGIN_NO_SERIAL_INIT(SiPixelGainCalibrationForHLTSimRcd,
                               SiPixelGainCalibrationForHLT,
                               InitGains<SiPixelGainCalibrationForHLT>);
REGISTER_PLUGIN_NO_SERIAL_INIT(SiPixelGainCalibrationOfflineSimRcd,
                               SiPixelGainCalibrationOffline,
                               InitGains<SiPixelGainCalibrationOffline>);
REGISTER_PLUGIN(SiPixelLorentzAngleRcd, SiPixelLorentzAngle);
REGISTER_PLUGIN_NO_SERIAL(SiPixelLorentzAngleSimRcd, SiPixelLorentzAngle);
REGISTER_PLUGIN(SiPixelVCalRcd, SiPixelVCal);
REGISTER_PLUGIN_NO_SERIAL(SiPixelVCalSimRcd, SiPixelVCal);
REGISTER_PLUGIN(SiPixelDynamicInefficiencyRcd, SiPixelDynamicInefficiency);
REGISTER_PLUGIN(SiPixelCalibConfigurationRcd, SiPixelCalibConfiguration);
REGISTER_PLUGIN(SiPixelPerformanceSummaryRcd, SiPixelPerformanceSummary);
REGISTER_PLUGIN(SiPixelQualityFromDbRcd, SiPixelQuality);
REGISTER_PLUGIN(SiPixelStatusScenariosRcd, SiPixelFEDChannelContainer);
REGISTER_PLUGIN(SiPixelStatusScenarioProbabilityRcd, SiPixelQualityProbabilities);
REGISTER_PLUGIN(SiPixelCPEGenericErrorParmRcd, SiPixelCPEGenericErrorParm);
REGISTER_PLUGIN(SiPixelTemplateDBObjectRcd, SiPixelTemplateDBObject);
REGISTER_PLUGIN(SiPixel2DTemplateDBObjectRcd, SiPixel2DTemplateDBObject);
REGISTER_PLUGIN(SiPixelGenErrorDBObjectRcd, SiPixelGenErrorDBObject);

REGISTER_PLUGIN(PixelCaenChannelIsOnRcd, PixelDCSObject<bool>);
REGISTER_PLUGIN(PixelCaenChannelIMonRcd, PixelDCSObject<float>);
REGISTER_PLUGIN(PixelCaenChannelRcd, PixelDCSObject<CaenChannel>);

REGISTER_PLUGIN(SiPixelDetVOffRcd, SiStripDetVOff);
