#ifndef RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCorrectionDeviceCollection_h
#define RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCorrectionDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalCMCorrectionSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using HGCalCMCorrectionDeviceCollection = PortableCollection<hgcalcmml::HGCalCMCorrectionSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCorrectionDeviceCollection_h
