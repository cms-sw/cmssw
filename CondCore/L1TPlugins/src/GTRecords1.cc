#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"

REGISTER_PLUGIN(L1GtBoardMapsRcd, L1GtBoardMaps);

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

REGISTER_PLUGIN(L1GtParametersRcd, L1GtParameters);

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

REGISTER_PLUGIN(L1GtPrescaleFactorsAlgoTrigRcd, L1GtPrescaleFactors);
REGISTER_PLUGIN_NO_SERIAL(L1GtPrescaleFactorsTechTrigRcd, L1GtPrescaleFactors);

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

REGISTER_PLUGIN(L1GtStableParametersRcd, L1GtStableParameters);
