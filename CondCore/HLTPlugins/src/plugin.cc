#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"

#include "CondFormats/HLTObjects/interface/HLTPrescaleTableCond.h"
#include "CondFormats/DataRecord/interface/HLTPrescaleTableRcd.h"

REGISTER_PLUGIN(AlCaRecoTriggerBitsRcd, AlCaRecoTriggerBits);
REGISTER_PLUGIN(HLTPrescaleTableRcd, trigger::HLTPrescaleTableCond);
