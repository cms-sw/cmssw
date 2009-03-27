#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/DQMObjects/interface/DQMSummary.h"
#include "CondFormats/DataRecord/interface/DQMSummaryRcd.h"
DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(DQMSummaryRcd, DQMSummary);
