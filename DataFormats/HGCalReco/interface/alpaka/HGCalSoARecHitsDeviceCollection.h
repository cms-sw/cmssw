#ifndef DataFormats_PortableTestObjects_interface_alpaka_HGCalSoARecHitsDeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_HGCalSoARecHitsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // SoA with x, y, z, id fields in device global memory
  using HGCalSoARecHitsDeviceCollection = PortableCollection<HGCalSoARecHits>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_PortableTestObjects_interface_alpaka_HGCalSoARecHitsDeviceCollection_h
