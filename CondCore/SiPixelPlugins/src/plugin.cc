#include "CondCore/PluginSystem/interface/registration_macros.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(SiPixelFedCablingMapRcd,SiPixelFedCablingMap);
REGISTER_PLUGIN(SiPixelGainCalibrationRcd,SiPixelGainCalibration);
REGISTER_PLUGIN(SiPixelLorentzAngleRcd,SiPixelLorentzAngle);