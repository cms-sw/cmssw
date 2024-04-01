#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HGCalCalibrationParameterHostCollection_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HGCalCalibrationParameterHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalCalibrationParameterSoA.h"

namespace hgcalrechit {

  // SoA with channel-level calibration parameters in host memory:
  //   pedestal, CM_slope, CM_ped, BXm1_kappa
  using HGCalCalibParamHostCollection = PortableHostCollection<HGCalCalibParamSoA>;
  
  // SoA with ROC-level configuration parameters in host memory:
  //   gain
  using HGCalConfigParamHostCollection = PortableHostCollection<HGCalConfigParamSoA>;

}  // namespace hgcalrechit

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_HGCalCalibrationParameterHostCollection_h
