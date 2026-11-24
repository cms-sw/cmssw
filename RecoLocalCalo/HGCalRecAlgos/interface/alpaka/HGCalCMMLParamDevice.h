#ifndef RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMMLParamDevice_h
#define RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMMLParamDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalCMMLParamHost.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalSoACMML.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using HGCalCMMLModuleParamDevice = PortableCollection<hgcalcmml::HGCalCMMLModuleParamSoA>;
  using HGCalCMMLNormParamDevice = PortableCollection<hgcalcmml::HGCalCMMLNormParamSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMMLParamDevice_h
