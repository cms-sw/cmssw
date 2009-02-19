#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/L1TriggerScalerRcd.h"
#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"

#include "CondFormats/RunInfo/interface/LuminosityInfo.h"
#include "CondFormats/DataRecord/interface/LuminosityInfoRcd.h"
#include "CondFormats/RunInfo/interface/HLTScaler.h"
#include "CondFormats/DataRecord/interface/HLTScalerRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(RunSummaryRcd,RunSummary);
REGISTER_PLUGIN(RunInfoRcd,RunInfo);
REGISTER_PLUGIN(L1TriggerScalerRcd, L1TriggerScaler);
REGISTER_PLUGIN(LuminosityInfoRcd, lumi::LuminosityInfo);
REGISTER_PLUGIN(HLTScalerRcd, lumi::HLTScaler);
