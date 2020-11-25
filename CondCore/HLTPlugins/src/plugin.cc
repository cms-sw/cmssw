#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"

#include "CondFormats/HLTObjects/interface/HLTPrescaleTableCond.h"
#include "CondFormats/DataRecord/interface/HLTPrescaleTableRcd.h"

#include "CondFormats/HLTObjects/interface/L1TObjScalingConstants.h"
#include "CondFormats/DataRecord/interface/L1TObjScalingRcd.h"

REGISTER_PLUGIN(AlCaRecoTriggerBitsRcd, AlCaRecoTriggerBits);
REGISTER_PLUGIN(HLTPrescaleTableRcd, trigger::HLTPrescaleTableCond);
REGISTER_PLUGIN(L1TObjScalingRcd, L1TObjScalingConstants);
