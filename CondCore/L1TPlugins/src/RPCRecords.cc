#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"

REGISTER_PLUGIN(L1RPCConfigRcd, L1RPCConfig);

#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"
#include "CondFormats/DataRecord/interface/L1RPCConeDefinitionRcd.h"

REGISTER_PLUGIN(L1RPCConeDefinitionRcd, L1RPCConeDefinition);

#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCHsbConfigRcd.h"

REGISTER_PLUGIN(L1RPCHsbConfigRcd, L1RPCHsbConfig);

#include "CondFormats/L1TObjects/interface/L1RPCBxOrConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCBxOrConfigRcd.h"

REGISTER_PLUGIN(L1RPCBxOrConfigRcd, L1RPCBxOrConfig);
