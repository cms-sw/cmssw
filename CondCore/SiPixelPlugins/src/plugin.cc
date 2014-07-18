#include "CondCore/ESSources/interface/registration_macros.h"

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
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
//#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"
#include "CondFormats/DataRecord/interface/SiPixelCPEGenericErrorParmRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObject38TRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObject4TRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObject0TRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"
#include "CondFormats/DataRecord/interface/PixelDCSRcds.h"

#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationOfflineSimRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTSimRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"

namespace {
 struct InitRocs {void operator()(SiPixelFedCablingMap& m){ m.initializeRocs();}};
 template<typename G> struct InitGains {void operator()(G& g){ g.initialize();}}; 
}


REGISTER_PLUGIN_INIT(SiPixelFedCablingMapRcd,SiPixelFedCablingMap, InitRocs);
REGISTER_PLUGIN_INIT(SiPixelGainCalibrationRcd,SiPixelGainCalibration, InitGains<SiPixelGainCalibration>);
REGISTER_PLUGIN_INIT(SiPixelGainCalibrationForHLTRcd,SiPixelGainCalibrationForHLT, InitGains<SiPixelGainCalibrationForHLT>);
REGISTER_PLUGIN_INIT(SiPixelGainCalibrationOfflineRcd,SiPixelGainCalibrationOffline, InitGains<SiPixelGainCalibrationOffline>);
REGISTER_PLUGIN_INIT(SiPixelGainCalibrationForHLTSimRcd,SiPixelGainCalibrationForHLT, InitGains<SiPixelGainCalibrationForHLT>);
REGISTER_PLUGIN_INIT(SiPixelGainCalibrationOfflineSimRcd,SiPixelGainCalibrationOffline, InitGains<SiPixelGainCalibrationOffline>);
REGISTER_PLUGIN(SiPixelLorentzAngleRcd,SiPixelLorentzAngle);
REGISTER_PLUGIN(SiPixelLorentzAngleSimRcd,SiPixelLorentzAngle);
REGISTER_PLUGIN(SiPixelCalibConfigurationRcd,SiPixelCalibConfiguration);
REGISTER_PLUGIN(SiPixelPerformanceSummaryRcd,SiPixelPerformanceSummary);
REGISTER_PLUGIN(SiPixelQualityFromDbRcd,SiPixelQuality);
REGISTER_PLUGIN(SiPixelCPEGenericErrorParmRcd,SiPixelCPEGenericErrorParm);
REGISTER_PLUGIN(SiPixelTemplateDBObjectRcd,SiPixelTemplateDBObject);
REGISTER_PLUGIN(SiPixelTemplateDBObject38TRcd,SiPixelTemplateDBObject);
REGISTER_PLUGIN(SiPixelTemplateDBObject4TRcd,SiPixelTemplateDBObject);
REGISTER_PLUGIN(SiPixelTemplateDBObject0TRcd,SiPixelTemplateDBObject);
REGISTER_PLUGIN(SiPixelGenErrorDBObjectRcd,SiPixelGenErrorDBObject);

REGISTER_PLUGIN(PixelCaenChannelIsOnRcd, PixelDCSObject<bool>);
REGISTER_PLUGIN(PixelCaenChannelIMonRcd, PixelDCSObject<float>);
REGISTER_PLUGIN(PixelCaenChannelRcd, PixelDCSObject<CaenChannel>);

REGISTER_PLUGIN(SiPixelDetVOffRcd,SiStripDetVOff);
