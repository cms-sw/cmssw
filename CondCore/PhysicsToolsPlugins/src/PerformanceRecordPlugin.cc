#include "CondCore/PluginSystem/interface/registration_macros.h"

#include "CondFormats/DataRecord/interface/PerformancePayloadRecord.h"
#include "CondFormats/DataRecord/interface/PerformanceWPRecord.h"

#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayload.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"

DEFINE_SEAL_MODULE();

REGISTER_PLUGIN(PerformancePayloadRecord, PerformancePayload);
REGISTER_PLUGIN(PerformanceWPRecord, PerformanceWorkingPoint);
