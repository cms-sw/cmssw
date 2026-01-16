#ifndef CondFormats_HGCalObjects_interface_HGCalCalibrationParameterHost_h
#define CondFormats_HGCalObjects_interface_HGCalCalibrationParameterHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibParamSoA.h"

namespace hgcalrechit {

  // SoA with channel-level calibration parameters in host memory:
  //   pedestal, CM_slope, CM_ped, BXm1_kappa
  using HGCalCalibParamHost = PortableHostCollection<HGCalCalibParamSoA>;

}  // namespace hgcalrechit

#endif  // CondFormats_HGCalObjects_interface_HGCalCalibrationParameterHost_h
