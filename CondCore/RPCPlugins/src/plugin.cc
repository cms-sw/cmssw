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
#include "CondFormats/RPCObjects/interface/L1RPCConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(RPCReadOutMappingRcd,RPCReadOutMapping);
REGISTER_PLUGIN(L1RPCConfigRcd,L1RPCConfig);
REGISTER_PLUGIN(RPCEMapRcd,RPCEMap);
