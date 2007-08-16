/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Paolo Ronchese, 2005 Nov 14,
 *    based on template by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
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


DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(DTReadOutMappingRcd,DTReadOutMapping);
REGISTER_PLUGIN(DTT0Rcd,DTT0);
REGISTER_PLUGIN(DTRangeT0Rcd,DTRangeT0);
REGISTER_PLUGIN(DTTtrigRcd,DTTtrig);
REGISTER_PLUGIN(DTMtimeRcd,DTMtime);
REGISTER_PLUGIN(DTStatusFlagRcd,DTStatusFlag);
REGISTER_PLUGIN(DTDeadFlagRcd,DTDeadFlag);
REGISTER_PLUGIN(DTPerformanceRcd,DTPerformance);
