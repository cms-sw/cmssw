#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondFormats/DataRecord/interface/LumiSectionDataRcd.h"
#include "CondFormats/Luminosity/interface/LumiCorrections.h"
#include "CondFormats/DataRecord/interface/LumiCorrectionsRcd.h"


REGISTER_PLUGIN(LumiSectionDataRcd, lumi::LumiSectionData);
REGISTER_PLUGIN(LumiCorrectionsRcd, LumiCorrections);

