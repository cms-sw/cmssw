/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CondFormats/CSCObjects/interface/CSCIdentifier.h"
#include "CondFormats/DataRecord/interface/CSCIdentifierRcd.h"
#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CondFormats/CSCObjects/interface/CSCReadoutMapping.h"
#include "CondFormats/DataRecord/interface/CSCReadoutMappingRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(CSCPedestalsRcd,CSCPedestals);
REGISTER_PLUGIN(CSCDBPedestalsRcd,CSCDBPedestals);
REGISTER_PLUGIN(CSCGainsRcd,CSCGains);
REGISTER_PLUGIN(CSCDBGainsRcd,CSCDBGains);
REGISTER_PLUGIN(CSCcrosstalkRcd,CSCcrosstalk);
REGISTER_PLUGIN(CSCDBCrosstalkRcd,CSCDBCrosstalk);
REGISTER_PLUGIN(CSCNoiseMatrixRcd,CSCNoiseMatrix);
REGISTER_PLUGIN(CSCDBNoiseMatrixRcd,CSCDBNoiseMatrix);
REGISTER_PLUGIN(CSCIdentifierRcd,CSCIdentifier);
REGISTER_PLUGIN(CSCReadoutMappingRcd,CSCReadoutMapping);
