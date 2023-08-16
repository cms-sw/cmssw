#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"

REGISTER_PLUGIN(L1JetEtScaleRcd, L1CaloEtScale);
REGISTER_PLUGIN_NO_SERIAL(L1EmEtScaleRcd, L1CaloEtScale);
REGISTER_PLUGIN_NO_SERIAL(L1HtMissScaleRcd, L1CaloEtScale);
REGISTER_PLUGIN_NO_SERIAL(L1HfRingEtScaleRcd, L1CaloEtScale);

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"

REGISTER_PLUGIN(L1MuTriggerScalesRcd, L1MuTriggerScales);

#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

REGISTER_PLUGIN(L1MuTriggerPtScaleRcd, L1MuTriggerPtScale);

#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/DataRecord/interface/L1MuGMTScalesRcd.h"

REGISTER_PLUGIN(L1MuGMTScalesRcd, L1MuGMTScales);
