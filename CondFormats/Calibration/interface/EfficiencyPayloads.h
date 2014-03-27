#include "CondFormats/Calibration/interface/Efficiency.h"
#include "CondCore/CondDB/interface/Serialization.h"

namespace cond {
  // payload factory specializations
  template <> condex::Efficiency* createPayload<condex::Efficiency>( const std::string& payloadTypeName ){
    if( payloadTypeName == "condex::ParametricEfficiencyInPt" ) return new condex::ParametricEfficiencyInPt;
    if( payloadTypeName == "condex::ParametricEfficiencyInEta" ) return new condex::ParametricEfficiencyInEta;
    throwException(std::string("Type mismatch, target object is type \"")+payloadTypeName+"\"",
		   "createPayload" );
  }
}

