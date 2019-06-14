#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFAlignment.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCPtLut.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/L1MuCSCPtLutRcd.h"

REGISTER_PLUGIN(L1MuCSCTFConfigurationRcd, L1MuCSCTFConfiguration);
REGISTER_PLUGIN(L1MuCSCTFAlignmentRcd, L1MuCSCTFAlignment);
REGISTER_PLUGIN(L1MuCSCPtLutRcd, L1MuCSCPtLut);
