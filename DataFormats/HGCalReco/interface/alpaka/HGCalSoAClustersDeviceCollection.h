#ifndef DataFormats_PortableTestObjects_interface_alpaka_HGCalSoAClustersDeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_HGCalSoAClustersDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoAClusters.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using HGCalSoAClustersDeviceCollection = PortableCollection<HGCalSoAClusters>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_PortableTestObjects_interface_alpaka_HGCalSoAClustersDeviceCollection_h
