#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/DataRecord/interface/RunNumberRcd.h"
#include "CondFormats/RunInfo/interface/RunNumber.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/L1TriggerScalerRcd.h"
#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"

using namespace runinfo_test;

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(RunNumberRcd,RunNumber);
REGISTER_PLUGIN(RunSummaryRcd,RunSummary);
REGISTER_PLUGIN(L1TriggerScalerRcd, L1TriggerScaler);
