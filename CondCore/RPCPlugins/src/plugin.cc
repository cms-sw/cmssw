/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Marcello Maggi --INFN
 *    based on template by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/DataRecord/interface/RPCReadOutMappingRcd.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"
#include "CondFormats/RPCObjects/interface/RPCMaskedStrips.h"
#include "CondFormats/DataRecord/interface/RPCMaskedStripsRcd.h"
#include "CondFormats/RPCObjects/interface/RPCDeadStrips.h"
#include "CondFormats/DataRecord/interface/RPCDeadStripsRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCHwConfigRcd.h"


#include "CondFormats/DataRecord/interface/RPCObGasRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObGas.h"
#include "CondFormats/DataRecord/interface/RPCObCondRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObCond.h"

#include "CondFormats/DataRecord/interface/RPCObPVSSmapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObPVSSmap.h"

#include "CondFormats/DataRecord/interface/RPCObFebmapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObFebmap.h"

#include "CondFormats/DataRecord/interface/RPCObGasmapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObGasmap.h"

#include "CondFormats/DataRecord/interface/RBCBoardSpecsRcd.h"
#include "CondFormats/DataRecord/interface/TTUBoardSpecsRcd.h"
#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"
#include "CondFormats/RPCObjects/interface/TTUBoardSpecs.h"


DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(RPCReadOutMappingRcd,RPCReadOutMapping);
REGISTER_PLUGIN(RPCEMapRcd,RPCEMap);
REGISTER_PLUGIN(RPCStripNoisesRcd,RPCStripNoises);
REGISTER_PLUGIN(L1RPCHwConfigRcd,L1RPCHwConfig);
REGISTER_PLUGIN(RPCObGasRcd,RPCObGas);
REGISTER_PLUGIN(RPCObImonRcd,RPCObImon);
REGISTER_PLUGIN(RPCObVmonRcd,RPCObVmon);
REGISTER_PLUGIN(RPCObStatusRcd,RPCObStatus);
REGISTER_PLUGIN(RPCObTempRcd,RPCObTemp);
REGISTER_PLUGIN(RPCObPVSSmapRcd,RPCObPVSSmap);
REGISTER_PLUGIN(RPCMaskedStripsRcd, RPCMaskedStrips);
REGISTER_PLUGIN(RPCDeadStripsRcd, RPCDeadStrips);
REGISTER_PLUGIN(RPCObFebmapRcd, RPCObFebmap);
REGISTER_PLUGIN(RPCObGasmapRcd, RPCObGasmap);

REGISTER_PLUGIN(RBCBoardSpecsRcd,RBCBoardSpecs);
REGISTER_PLUGIN(TTUBoardSpecsRcd,TTUBoardSpecs);
