#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/GEMObjects/interface/GEMEMap.h"
#include "CondFormats/DataRecord/interface/GEMEMapRcd.h"
#include "CondFormats/GEMObjects/interface/ME0EMap.h"
#include "CondFormats/DataRecord/interface/ME0EMapRcd.h"

REGISTER_PLUGIN(GEMEMapRcd,GEMEMap); 
REGISTER_PLUGIN(ME0EMapRcd,ME0EMap); 
