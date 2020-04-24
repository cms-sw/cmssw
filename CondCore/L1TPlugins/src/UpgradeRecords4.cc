#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/L1TObjects/interface/L1TUtmScale.h"
#include "CondFormats/DataRecord/interface/L1TUtmScaleRcd.h"
REGISTER_PLUGIN(L1TUtmScaleRcd, L1TUtmScale);

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuO2ORcd.h"
REGISTER_PLUGIN(L1TUtmTriggerMenuRcd, L1TUtmTriggerMenu);
REGISTER_PLUGIN(L1TUtmTriggerMenuO2ORcd, L1TUtmTriggerMenu);

#include "CondFormats/L1TObjects/interface/L1TGlobalParameters.h"
#include "CondFormats/DataRecord/interface/L1TGlobalParametersRcd.h"
REGISTER_PLUGIN(L1TGlobalParametersRcd, L1TGlobalParameters);

#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetos.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosRcd.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosO2ORcd.h"
REGISTER_PLUGIN(L1TGlobalPrescalesVetosRcd, L1TGlobalPrescalesVetos);
REGISTER_PLUGIN(L1TGlobalPrescalesVetosO2ORcd, L1TGlobalPrescalesVetos);
