#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"

REGISTER_PLUGIN(L1TriggerKeyRcd, L1TriggerKey);

#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"

REGISTER_PLUGIN(L1TriggerKeyExtRcd, L1TriggerKeyExt);

#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"

REGISTER_PLUGIN(L1TriggerKeyListRcd, L1TriggerKeyList);

#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"

REGISTER_PLUGIN(L1TriggerKeyListExtRcd, L1TriggerKeyListExt);
