#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/DataRecord/interface/PerformancePayloadRecord.h"
#include "CondFormats/DataRecord/interface/PerformanceWPRecord.h"
#include "CondFormats/DataRecord/interface/PFCalibrationRcd.h"

#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayload.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"

#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromBinnedTFormula.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTable.h"

#include "CondCore/CondDB/interface/Serialization.h"

namespace cond {
  template <> std::unique_ptr<PerformancePayload> deserialize<PerformancePayload>( const std::string& payloadType,
                                                                                   const Binary& payloadData,
                                                                                   const Binary& streamerInfoData ){
    // DESERIALIZE_BASE_CASE( PerformancePayload );  abstract 
    DESERIALIZE_POLIMORPHIC_CASE( PerformancePayload, PerformancePayloadFromTFormula ); 
    DESERIALIZE_POLIMORPHIC_CASE( PerformancePayload, PerformancePayloadFromBinnedTFormula ); 
    DESERIALIZE_POLIMORPHIC_CASE( PerformancePayload, PerformancePayloadFromTable ); 
    // here we come if none of the deserializations above match the payload type:
    throwException(std::string("Type mismatch, target object is type \"")+payloadType+"\"",
		   "createPayload" );
  }
}

namespace {
  struct InitPerformancePayload {void operator()(PerformancePayload& e){ e.initialize();}};
}

REGISTER_PLUGIN_INIT(PerformancePayloadRecord, PerformancePayload, InitPerformancePayload);
REGISTER_PLUGIN(PerformanceWPRecord, PerformanceWorkingPoint);
REGISTER_PLUGIN_INIT(PFCalibrationRcd, PerformancePayload, InitPerformancePayload);
