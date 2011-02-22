#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/DataRecord/interface/PerformancePayloadRecord.h"
#include "CondFormats/DataRecord/interface/PerformanceWPRecord.h"
#include "CondFormats/DataRecord/interface/PFCalibrationRcd.h"

#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayload.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"



REGISTER_PLUGIN(PerformancePayloadRecord, PerformancePayload);
REGISTER_PLUGIN(PerformanceWPRecord, PerformanceWorkingPoint);
REGISTER_PLUGIN(PFCalibrationRcd, PerformancePayload);
