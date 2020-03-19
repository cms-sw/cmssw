#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTParameters.h"
#include "CondFormats/DataRecord/interface/L1MuGMTParametersRcd.h"

REGISTER_PLUGIN(L1MuGMTParametersRcd, L1MuGMTParameters);

#include "CondFormats/L1TObjects/interface/L1MuGMTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1MuGMTChannelMaskRcd.h"

REGISTER_PLUGIN(L1MuGMTChannelMaskRcd, L1MuGMTChannelMask);
