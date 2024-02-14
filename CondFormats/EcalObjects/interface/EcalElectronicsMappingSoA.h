#ifndef CondFormats_EcalObjects_EcalElectronicsMappingSoA_h
#define CondFormats_EcalObjects_EcalElectronicsMappingSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

GENERATE_SOA_LAYOUT(EcalElectronicsMappingSoALayout, SOA_COLUMN(uint32_t, rawid))

using EcalElectronicsMappingSoA = EcalElectronicsMappingSoALayout<>;

#endif
