#include "HeterogeneousCore/AlpakaCore/interface/alpaka/typelookup.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCalibrationParameterDeviceCollection.h"

//TYPELOOKUP_ALPAKA_DATA_REG(hgcalrechit::HGCalCalibParamHostCollection); // redundant ?
//TYPELOOKUP_ALPAKA_DATA_REG(hgcalrechit::HGCalConfigParamHostCollection); // redundant ?
TYPELOOKUP_ALPAKA_DATA_REG(hgcalrechit::HGCalCalibParamDeviceCollection);
TYPELOOKUP_ALPAKA_DATA_REG(hgcalrechit::HGCalConfigParamDeviceCollection);
