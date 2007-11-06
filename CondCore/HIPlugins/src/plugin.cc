#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(HeavyIonRcd,CentralityTable);
