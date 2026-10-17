#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HGCALSoACMMLDeviceCollection_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HGCALSoACMMLDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCALSoACMML.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using HGCalSoACMMLDeviceCollection = PortableCollection<hgcalcmml::HGCalCMMLSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_HGCALSoACMMLDeviceCollection_h
