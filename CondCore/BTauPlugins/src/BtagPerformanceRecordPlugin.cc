#include "CondCore/PluginSystem/interface/registration_macros.h"

#include "CondFormats/DataRecord/interface/BTagPerformancePayloadRecord.h"
#include "CondFormats/DataRecord/interface/BTagPerformanceWPRecord.h"

#include "CondFormats/BTauObjects/interface/BtagPerformancePayload.h"
#include "CondFormats/BTauObjects/interface/BtagWorkingPoint.h"

DEFINE_SEAL_MODULE();

REGISTER_PLUGIN(BTagPerformancePayloadRecord, BtagPerformancePayload);
REGISTER_PLUGIN(BTagPerformanceWPRecord, BtagWorkingPoint);
