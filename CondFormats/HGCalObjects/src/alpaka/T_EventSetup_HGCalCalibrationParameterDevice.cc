#include "HeterogeneousCore/AlpakaCore/interface/alpaka/typelookup.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"

//TYPELOOKUP_ALPAKA_DATA_REG(hgcalrechit::HGCalCalibParamHost); // redundant ?
//TYPELOOKUP_ALPAKA_DATA_REG(hgcalrechit::HGCalConfigParamHost); // redundant ?
TYPELOOKUP_ALPAKA_DATA_REG(hgcalrechit::HGCalCalibParamDevice);
TYPELOOKUP_ALPAKA_DATA_REG(hgcalrechit::HGCalConfigParamDevice);
