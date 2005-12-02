/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "CondFormats/DataRecord/interface/CSCPedestalsRcd.h"
#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"
#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"
#include "CondFormats/CSCObjects/interface/CSCIdentifier.h"
#include "CondFormats/DataRecord/interface/CSCIdentifierRcd.h"
#include "CondFormats/CSCObjects/interface/CSCReadoutMapping.h"
#include "CondFormats/DataRecord/interface/CSCReadoutMappinRcd.h"
DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(CSCPedestalsRcd,CSCPedestals);
REGISTER_PLUGIN(CSCGainsRcd,CSCGains);
REGISTER_PLUGIN(CSCcrosstalkRcd,CSCcrosstalk);
REGISTER_PLUGIN(CSCIdentifierRcd,CSCIdentifier);
REGISTER_PLUGIN(CSCReadoutMappingRcd,CSCReadoutMapping);
