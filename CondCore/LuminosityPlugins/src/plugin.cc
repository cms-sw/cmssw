#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondFormats/DataRecord/interface/LumiSectionDataRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(LumiSectionDataRcd, lumi::LumiSectionData);



