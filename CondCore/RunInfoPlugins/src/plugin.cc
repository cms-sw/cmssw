#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/L1TriggerScalerRcd.h"
#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"


REGISTER_PLUGIN(RunSummaryRcd,RunSummary);
REGISTER_PLUGIN(RunInfoRcd,RunInfo);
REGISTER_PLUGIN(L1TriggerScalerRcd, L1TriggerScaler);
