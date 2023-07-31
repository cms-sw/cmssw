#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"

REGISTER_PLUGIN(L1RCTParametersRcd, L1RCTParameters);

#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"

REGISTER_PLUGIN(L1RCTChannelMaskRcd, L1RCTChannelMask);

#include "CondFormats/L1TObjects/interface/L1RCTNoisyChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTNoisyChannelMaskRcd.h"

REGISTER_PLUGIN(L1RCTNoisyChannelMaskRcd, L1RCTNoisyChannelMask);

#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"

REGISTER_PLUGIN(L1CaloEcalScaleRcd, L1CaloEcalScale);

#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"

//L1CaloHcalScale is same as L1RCTParameters
REGISTER_PLUGIN_NO_SERIAL(L1CaloHcalScaleRcd, L1CaloHcalScale);
