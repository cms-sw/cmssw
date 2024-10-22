#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsRcd_deprecated.h"  // Temporary copy to avoid crashes - AWB 28.08.17
// Required to satisfy current convention for "Record" in "Global Tag Entries"
// https://cms-conddb-prod.cern.ch/cmsDbBrowser/search/Prod/L1TMuonEndCapParams

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsO2ORcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestO2ORcd.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapFwVersion.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionO2ORcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsO2ORcd.h"

#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsO2ORcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelKalmanParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelKalmanParamsRcd.h"

#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsO2ORcd.h"

REGISTER_PLUGIN(L1TMuonEndcapParamsRcd, L1TMuonEndCapParams);  // Temporary copy to avoid crashes - AWB 28.08.17

REGISTER_PLUGIN_NO_SERIAL(L1TMuonEndCapParamsRcd, L1TMuonEndCapParams);
REGISTER_PLUGIN(L1TMuonEndCapForestRcd, L1TMuonEndCapForest);
REGISTER_PLUGIN(L1TMuonOverlapFwVersionRcd, L1TMuonOverlapFwVersion);
REGISTER_PLUGIN(L1TMuonOverlapParamsRcd, L1TMuonOverlapParams);
REGISTER_PLUGIN(L1TMuonBarrelParamsRcd, L1TMuonBarrelParams);
REGISTER_PLUGIN(L1TMuonBarrelKalmanParamsRcd, L1TMuonBarrelKalmanParams);
REGISTER_PLUGIN(L1TMuonGlobalParamsRcd, L1TMuonGlobalParams);

REGISTER_PLUGIN_NO_SERIAL(L1TMuonEndCapParamsO2ORcd, L1TMuonEndCapParams);
REGISTER_PLUGIN_NO_SERIAL(L1TMuonEndCapForestO2ORcd, L1TMuonEndCapForest);
REGISTER_PLUGIN_NO_SERIAL(L1TMuonOverlapFwVersionO2ORcd, L1TMuonOverlapFwVersion);
REGISTER_PLUGIN_NO_SERIAL(L1TMuonOverlapParamsO2ORcd, L1TMuonOverlapParams);
REGISTER_PLUGIN_NO_SERIAL(L1TMuonBarrelParamsO2ORcd, L1TMuonBarrelParams);
REGISTER_PLUGIN_NO_SERIAL(L1TMuonGlobalParamsO2ORcd, L1TMuonGlobalParams);
