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
DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(RPCReadOutMappingRcd,RPCReadOutMapping);
