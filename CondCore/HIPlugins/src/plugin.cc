#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "CondFormats/DataRecord/interface/HeavyIonRPRcd.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondFormats/HIObjects/interface/RPFlatParams.h"

REGISTER_PLUGIN(HeavyIonRPRcd,RPFlatParams);
REGISTER_PLUGIN(HeavyIonRcd,CentralityTable);
