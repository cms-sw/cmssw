#ifndef CondFormats_HGCalObjects_interface_HGCalCalibrationParameterHost_h
#define CondFormats_HGCalObjects_interface_HGCalCalibrationParameterHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibParamSoA.h"
#include "CondFormats/HGCalObjects/interface/HGCalConfigParamSoA.h"

namespace hgcalrechit {

  // SoA with channel-level calibration parameters in host memory:
  //   pedestal, CM_slope, CM_ped, BXm1_kappa
  using HGCalCalibParamHost = PortableHostCollection<HGCalCalibParamSoA>;

  // SoA with ROC-level configuration parameters in host memory:
  //   gain
  using HGCalConfigParamHost = PortableHostCollection<HGCalConfigParamSoA>;

}  // namespace hgcalrechit

#endif  // CondFormats_HGCalObjects_interface_HGCalCalibrationParameterHost_h
