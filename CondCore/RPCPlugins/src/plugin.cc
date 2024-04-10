/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Marcello Maggi --INFN
 *    based on template by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/DataRecord/interface/RPCReadOutMappingRcd.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"
#include "CondFormats/RPCObjects/interface/RPCClusterSize.h"
#include "CondFormats/DataRecord/interface/RPCClusterSizeRcd.h"
#include "CondFormats/RPCObjects/interface/RPCDQMObject.h"
#include "CondFormats/DataRecord/interface/RPCDQMObjectRcd.h"
#include "CondFormats/RPCObjects/interface/RPCMaskedStrips.h"
#include "CondFormats/DataRecord/interface/RPCMaskedStripsRcd.h"
#include "CondFormats/RPCObjects/interface/RPCDeadStrips.h"
#include "CondFormats/DataRecord/interface/RPCDeadStripsRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCHwConfigRcd.h"

#include "CondFormats/DataRecord/interface/RPCObGasMixRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObGasMix.h"
#include "CondFormats/DataRecord/interface/RPCObGasHumRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObGasHum.h"
#include "CondFormats/DataRecord/interface/RPCObGasRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObGas.h"
#include "CondFormats/DataRecord/interface/RPCObCondRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/DataRecord/interface/RPCObUXCRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObUXC.h"

#include "CondFormats/DataRecord/interface/RPCObPVSSmapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCObPVSSmap.h"

#include "CondFormats/DataRecord/interface/RBCBoardSpecsRcd.h"
#include "CondFormats/DataRecord/interface/TTUBoardSpecsRcd.h"
#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"
#include "CondFormats/RPCObjects/interface/TTUBoardSpecs.h"

REGISTER_PLUGIN(RPCReadOutMappingRcd, RPCReadOutMapping);
REGISTER_PLUGIN(RPCEMapRcd, RPCEMap);
REGISTER_PLUGIN(RPCStripNoisesRcd, RPCStripNoises);
REGISTER_PLUGIN(RPCClusterSizeRcd, RPCClusterSize);  //new plugin 21.XII.2009 for RPC cluster size chamber by chamber
REGISTER_PLUGIN(RPCDQMObjectRcd, RPCDQMObject);      //new plugin for RPC DQMPVT
REGISTER_PLUGIN(L1RPCHwConfigRcd, L1RPCHwConfig);
REGISTER_PLUGIN(RPCObGasRcd, RPCObGas);
REGISTER_PLUGIN(RPCObImonRcd, RPCObImon);
REGISTER_PLUGIN(RPCObVmonRcd, RPCObVmon);
REGISTER_PLUGIN(RPCObStatusRcd, RPCObStatus);
REGISTER_PLUGIN(RPCObTempRcd, RPCObTemp);
REGISTER_PLUGIN(RPCObPVSSmapRcd, RPCObPVSSmap);
REGISTER_PLUGIN(RPCMaskedStripsRcd, RPCMaskedStrips);
REGISTER_PLUGIN(RPCDeadStripsRcd, RPCDeadStrips);
REGISTER_PLUGIN(RPCObUXCRcd, RPCObUXC);
REGISTER_PLUGIN(RBCBoardSpecsRcd, RBCBoardSpecs);
REGISTER_PLUGIN(TTUBoardSpecsRcd, TTUBoardSpecs);
REGISTER_PLUGIN(RPCObGasMixRcd, RPCObGasMix);
REGISTER_PLUGIN(RPCObGasHumRcd, RPCObGasHum);

#include "CondFormats/DataRecord/interface/RPCLBLinkMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCLBLinkMap.h"
REGISTER_PLUGIN(RPCLBLinkMapRcd, RPCLBLinkMap);

#include "CondFormats/DataRecord/interface/RPCDCCLinkMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCDCCLinkMap.h"
REGISTER_PLUGIN(RPCDCCLinkMapRcd, RPCDCCLinkMap);

#include "CondFormats/DataRecord/interface/RPCTwinMuxLinkMapRcd.h"
#include "CondFormats/DataRecord/interface/RPCOMTFLinkMapRcd.h"
#include "CondFormats/DataRecord/interface/RPCCPPFLinkMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"
REGISTER_PLUGIN(RPCTwinMuxLinkMapRcd, RPCAMCLinkMap);
REGISTER_PLUGIN_NO_SERIAL(RPCOMTFLinkMapRcd, RPCAMCLinkMap);
REGISTER_PLUGIN_NO_SERIAL(RPCCPPFLinkMapRcd, RPCAMCLinkMap);
