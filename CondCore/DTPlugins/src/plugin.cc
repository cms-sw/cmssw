/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Paolo Ronchese, 2005 Nov 14,
 *    based on template by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "CondFormats/DataRecord/interface/DTT0RefRcd.h"
#include "CondFormats/DTObjects/interface/DTRangeT0.h"
#include "CondFormats/DataRecord/interface/DTRangeT0Rcd.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTPerformance.h"
#include "CondFormats/DataRecord/interface/DTPerformanceRcd.h"
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DataRecord/interface/DTCCBConfigRcd.h"
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigContainerRcd.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"
#include "CondFormats/DataRecord/interface/DTTPGParametersRcd.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"
#include "CondFormats/DataRecord/interface/DTHVStatusRcd.h"
#include "CondFormats/DTObjects/interface/DTLVStatus.h"
#include "CondFormats/DataRecord/interface/DTLVStatusRcd.h"
#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondCore/CondDB/interface/KeyListProxy.h"


//
#include "CondCore/CondDB/interface/Serialization.h"

namespace cond {

  template <> BaseKeyed* createPayload<BaseKeyed>( const std::string& payloadTypeName ){
    if( payloadTypeName == "DTKeyedConfig" ) return new DTKeyedConfig;
    throwException(std::string("Type mismatch, target object is type \"")+payloadTypeName+"\"",
		   "createPayload" );
  }

}

REGISTER_PLUGIN(DTReadOutMappingRcd,DTReadOutMapping);
REGISTER_PLUGIN(DTT0Rcd,DTT0);
REGISTER_PLUGIN(DTT0RefRcd,DTT0);
REGISTER_PLUGIN(DTRangeT0Rcd,DTRangeT0);
REGISTER_PLUGIN(DTTtrigRcd,DTTtrig);
REGISTER_PLUGIN(DTMtimeRcd,DTMtime);
REGISTER_PLUGIN(DTStatusFlagRcd,DTStatusFlag);
REGISTER_PLUGIN(DTDeadFlagRcd,DTDeadFlag);
REGISTER_PLUGIN(DTPerformanceRcd,DTPerformance);
REGISTER_PLUGIN(DTCCBConfigRcd,DTCCBConfig);
REGISTER_PLUGIN(DTTPGParametersRcd,DTTPGParameters);
REGISTER_PLUGIN(DTHVStatusRcd,DTHVStatus);
REGISTER_PLUGIN(DTLVStatusRcd,DTLVStatus);
REGISTER_PLUGIN(DTKeyedConfigContainerRcd, cond::BaseKeyed);
REGISTER_KEYLIST_PLUGIN(DTKeyedConfigListRcd,cond::persistency::KeyList,DTKeyedConfigContainerRcd);



