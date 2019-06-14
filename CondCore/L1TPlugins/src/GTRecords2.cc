#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

REGISTER_PLUGIN(L1GtTriggerMaskAlgoTrigRcd, L1GtTriggerMask);
REGISTER_PLUGIN(L1GtTriggerMaskTechTrigRcd, L1GtTriggerMask);
REGISTER_PLUGIN(L1GtTriggerMaskVetoAlgoTrigRcd, L1GtTriggerMask);
REGISTER_PLUGIN(L1GtTriggerMaskVetoTechTrigRcd, L1GtTriggerMask);
