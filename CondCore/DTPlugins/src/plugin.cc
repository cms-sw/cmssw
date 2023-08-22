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
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"
#include "CondFormats/DataRecord/interface/DTTPGParametersRcd.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"
#include "CondFormats/DataRecord/interface/DTHVStatusRcd.h"
#include "CondFormats/DTObjects/interface/DTLVStatus.h"
#include "CondFormats/DataRecord/interface/DTLVStatusRcd.h"
#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondCore/CondDB/interface/KeyListProxy.h"
#include "CondFormats/DTObjects/interface/DTRecoUncertainties.h"
#include "CondFormats/DataRecord/interface/DTRecoUncertaintiesRcd.h"
#include "CondFormats/DTObjects/interface/DTRecoConditions.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsTtrigRcd.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsVdriftRcd.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsUncertRcd.h"

//
#include "CondCore/CondDB/interface/Serialization.h"
#include "CondFormats/External/interface/DetID.h"

#include <memory>

namespace {
  struct InitDTCCBConfig {
    void operator()(DTCCBConfig& e) { e.initialize(); }
  };
}  // namespace

namespace {
  struct InitDTDeadFlag {
    void operator()(DTDeadFlag& e) { e.initialize(); }
  };
}  // namespace

namespace {
  struct InitDTHVStatus {
    void operator()(DTHVStatus& e) { e.initialize(); }
  };
}  // namespace

namespace {
  struct InitDTLVStatus {
    void operator()(DTLVStatus& e) { e.initialize(); }
  };
}  // namespace

namespace {
  struct InitDTMtime {
    void operator()(DTMtime& e) { e.initialize(); }
  };
}  // namespace

namespace {
  struct InitDTPerformance {
    void operator()(DTPerformance& e) { e.initialize(); }
  };
}  // namespace

namespace {
  struct InitDTRangeT0 {
    void operator()(DTRangeT0& e) { e.initialize(); }
  };
}  // namespace

namespace {
  struct InitDTStatusFlag {
    void operator()(DTStatusFlag& e) { e.initialize(); }
  };
}  // namespace

namespace {
  struct InitDTTPGParameters {
    void operator()(DTTPGParameters& e) { e.initialize(); }
  };
}  // namespace

namespace {
  struct InitDTTtrig {
    void operator()(DTTtrig& e) { e.initialize(); }
  };
}  // namespace

REGISTER_PLUGIN(DTReadOutMappingRcd, DTReadOutMapping);
REGISTER_PLUGIN(DTT0Rcd, DTT0);
REGISTER_PLUGIN_NO_SERIAL(DTT0RefRcd, DTT0);
REGISTER_PLUGIN_INIT(DTRangeT0Rcd, DTRangeT0, InitDTRangeT0);
REGISTER_PLUGIN_INIT(DTTtrigRcd, DTTtrig, InitDTTtrig);
REGISTER_PLUGIN_INIT(DTMtimeRcd, DTMtime, InitDTMtime);
REGISTER_PLUGIN_INIT(DTStatusFlagRcd, DTStatusFlag, InitDTStatusFlag);
REGISTER_PLUGIN_INIT(DTDeadFlagRcd, DTDeadFlag, InitDTDeadFlag);
REGISTER_PLUGIN_INIT(DTPerformanceRcd, DTPerformance, InitDTPerformance);
REGISTER_PLUGIN_INIT(DTCCBConfigRcd, DTCCBConfig, InitDTCCBConfig);
REGISTER_PLUGIN_INIT(DTTPGParametersRcd, DTTPGParameters, InitDTTPGParameters);
REGISTER_PLUGIN_INIT(DTHVStatusRcd, DTHVStatus, InitDTHVStatus);
REGISTER_PLUGIN_INIT(DTLVStatusRcd, DTLVStatus, InitDTLVStatus);
REGISTER_PLUGIN(DTRecoUncertaintiesRcd, DTRecoUncertainties);
//New flexyble payloads for ttrig, vdrift, uncertainty
REGISTER_PLUGIN(DTRecoConditionsTtrigRcd, DTRecoConditions);
REGISTER_PLUGIN_NO_SERIAL(DTRecoConditionsVdriftRcd, DTRecoConditions);
REGISTER_PLUGIN_NO_SERIAL(DTRecoConditionsUncertRcd, DTRecoConditions);
