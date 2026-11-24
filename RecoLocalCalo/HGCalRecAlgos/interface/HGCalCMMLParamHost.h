#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HGCalCMMLParamHost_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HGCalCMMLParamHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalSoACMML.h"

namespace hgcalcmml {

  using HGCalCMMLModuleParamHost = PortableHostCollection<HGCalCMMLModuleParamSoA>;
  using HGCalCMMLNormParamHost = PortableHostCollection<HGCalCMMLNormParamSoA>;

}  // namespace hgcalcmml

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_HGCalCMMLParamHost_h
