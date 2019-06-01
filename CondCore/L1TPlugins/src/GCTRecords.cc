#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"

REGISTER_PLUGIN(L1GctChannelMaskRcd, L1GctChannelMask);

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"

REGISTER_PLUGIN(L1GctJetFinderParamsRcd, L1GctJetFinderParams);
