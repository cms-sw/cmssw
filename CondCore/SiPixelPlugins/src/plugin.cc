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
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEParmErrors.h"
#include "CondFormats/DataRecord/interface/SiPixelCPEParmErrorsRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CondFormats/DataRecord/interface/SiPixelCalibConfigurationRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiPixelPerformanceSummaryRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDisabledModules.h"
#include "CondFormats/DataRecord/interface/SiPixelDisabledModulesRcd.h"


DEFINE_SEAL_MODULE();

REGISTER_PLUGIN(SiPixelFedCablingMapRcd,SiPixelFedCablingMap);
REGISTER_PLUGIN(SiPixelGainCalibrationRcd,SiPixelGainCalibration);
REGISTER_PLUGIN(SiPixelGainCalibrationForHLTRcd,SiPixelGainCalibrationForHLT);
REGISTER_PLUGIN(SiPixelGainCalibrationOfflineRcd,SiPixelGainCalibrationOffline);
REGISTER_PLUGIN(SiPixelLorentzAngleRcd,SiPixelLorentzAngle);
REGISTER_PLUGIN(SiPixelCPEParmErrorsRcd,SiPixelCPEParmErrors);
REGISTER_PLUGIN(SiPixelCalibConfigurationRcd,SiPixelCalibConfiguration);
REGISTER_PLUGIN(SiPixelPerformanceSummaryRcd,SiPixelPerformanceSummary);
REGISTER_PLUGIN(SiPixelDisabledModulesRcd,SiPixelDisabledModules);

