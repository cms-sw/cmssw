#include "CondCore/PluginSystem/interface/registration_macros.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(SiPixelFedCablingMapRcd,SiPixelFedCablingMap);
