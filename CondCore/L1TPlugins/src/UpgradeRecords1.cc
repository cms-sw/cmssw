#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TCaloStage2ParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsO2ORcd.h"
#include "CondFormats/L1TObjects/interface/CaloConfig.h"
#include "CondFormats/DataRecord/interface/L1TCaloConfigRcd.h"
using namespace l1t;
REGISTER_PLUGIN(L1TCaloParamsO2ORcd, CaloParams);
REGISTER_PLUGIN_NO_SERIAL(L1TCaloParamsRcd, CaloParams);
REGISTER_PLUGIN_NO_SERIAL(L1TCaloStage2ParamsRcd, CaloParams);
REGISTER_PLUGIN(L1TCaloConfigRcd, CaloConfig);
