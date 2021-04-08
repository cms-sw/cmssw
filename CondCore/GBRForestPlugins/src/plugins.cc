#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "CondFormats/GBRForest/interface/GBRForestD.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"

REGISTER_PLUGIN(GBRWrapperRcd, GBRForest);
REGISTER_PLUGIN(GBRDWrapperRcd, GBRForestD);
