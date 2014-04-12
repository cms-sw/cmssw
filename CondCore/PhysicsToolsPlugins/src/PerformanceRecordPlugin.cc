#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/DataRecord/interface/PerformancePayloadRecord.h"
#include "CondFormats/DataRecord/interface/PerformanceWPRecord.h"
#include "CondFormats/DataRecord/interface/PFCalibrationRcd.h"

#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayload.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"

#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromBinnedTFormula.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTable.h"

namespace cond {
  template <> PerformancePayload* createPayload<PerformancePayload>( const std::string& payloadTypeName ){
    if( payloadTypeName == "PerformancePayloadFromTFormula" ) return new PerformancePayloadFromTFormula;
    if( payloadTypeName == "PerformancePayloadFromBinnedTFormula" ) return new PerformancePayloadFromBinnedTFormula;
    if( payloadTypeName == "PerformancePayloadFromTable" ) return new PerformancePayloadFromTable;
    throwException(std::string("Type mismatch, target object is type \"")+payloadTypeName+"\"",
		   "createPayload" );
  }
}

namespace {
  struct InitPerformancePayload {void operator()(PerformancePayload& e){ e.initialize();}};
}

REGISTER_PLUGIN_INIT(PerformancePayloadRecord, PerformancePayload, InitPerformancePayload);
REGISTER_PLUGIN(PerformanceWPRecord, PerformanceWorkingPoint);
REGISTER_PLUGIN_INIT(PFCalibrationRcd, PerformancePayload, InitPerformancePayload);
