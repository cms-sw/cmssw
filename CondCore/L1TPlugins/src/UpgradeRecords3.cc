#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/L1TObjects/interface/L1TUtmAlgorithm.h"
#include "CondFormats/DataRecord/interface/L1TUtmAlgorithmRcd.h"
REGISTER_PLUGIN(L1TUtmAlgorithmRcd, L1TUtmAlgorithm);

#include "CondFormats/L1TObjects/interface/L1TUtmBin.h"
#include "CondFormats/DataRecord/interface/L1TUtmBinRcd.h"
REGISTER_PLUGIN(L1TUtmBinRcd, L1TUtmBin);

#include "CondFormats/L1TObjects/interface/L1TUtmCondition.h"
#include "CondFormats/DataRecord/interface/L1TUtmConditionRcd.h"
REGISTER_PLUGIN(L1TUtmConditionRcd, L1TUtmCondition);

#include "CondFormats/L1TObjects/interface/L1TUtmCut.h"
#include "CondFormats/DataRecord/interface/L1TUtmCutRcd.h"
REGISTER_PLUGIN(L1TUtmCutRcd, L1TUtmCut);

#include "CondFormats/L1TObjects/interface/L1TUtmCutValue.h"
#include "CondFormats/DataRecord/interface/L1TUtmCutValueRcd.h"
REGISTER_PLUGIN(L1TUtmCutValueRcd, L1TUtmCutValue);

#include "CondFormats/L1TObjects/interface/L1TUtmObject.h"
#include "CondFormats/DataRecord/interface/L1TUtmObjectRcd.h"
// L1TUtmAlgorithm is same as L1TUtmObject
REGISTER_PLUGIN_NO_SERIAL(L1TUtmObjectRcd, L1TUtmObject);
