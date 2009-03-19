#include "CondCore/PluginSystem/interface/registration_macros.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationOfflineRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CondFormats/DataRecord/interface/SiPixelCalibConfigurationRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiPixelPerformanceSummaryRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"
#include "CondFormats/DataRecord/interface/SiPixelCPEGenericErrorParmRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"
#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"
#include "CondFormats/DataRecord/interface/PixelDCSRcds.h"

#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationOfflineSimRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTSimRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"


REGISTER_PLUGIN(SiPixelFedCablingMapRcd,SiPixelFedCablingMap);
REGISTER_PLUGIN(SiPixelGainCalibrationRcd,SiPixelGainCalibration);
REGISTER_PLUGIN(SiPixelGainCalibrationForHLTRcd,SiPixelGainCalibrationForHLT);
REGISTER_PLUGIN(SiPixelGainCalibrationOfflineRcd,SiPixelGainCalibrationOffline);
REGISTER_PLUGIN(SiPixelGainCalibrationForHLTSimRcd,SiPixelGainCalibrationForHLT);
REGISTER_PLUGIN(SiPixelGainCalibrationOfflineSimRcd,SiPixelGainCalibrationOffline);
REGISTER_PLUGIN(SiPixelLorentzAngleRcd,SiPixelLorentzAngle);
REGISTER_PLUGIN(SiPixelLorentzAngleSimRcd,SiPixelLorentzAngle);
REGISTER_PLUGIN(SiPixelCalibConfigurationRcd,SiPixelCalibConfiguration);
REGISTER_PLUGIN(SiPixelPerformanceSummaryRcd,SiPixelPerformanceSummary);
REGISTER_PLUGIN(SiPixelQualityRcd,SiPixelQuality);
REGISTER_PLUGIN(SiPixelCPEGenericErrorParmRcd,SiPixelCPEGenericErrorParm);
REGISTER_PLUGIN(SiPixelTemplateDBObjectRcd,SiPixelTemplateDBObject);

REGISTER_PLUGIN(PixelCaenChannelIsOnRcd, PixelDCSObject<bool>);
REGISTER_PLUGIN(PixelCaenChannelIMonRcd, PixelDCSObject<float>);
REGISTER_PLUGIN(PixelCaenChannelRcd, PixelDCSObject<CaenChannel>);
