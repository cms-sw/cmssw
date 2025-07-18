#ifndef DataFormats_PortableTestObjects_interface_alpaka_HGCalSoARecHitsExtraDeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_HGCalSoARecHitsExtraDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsExtra.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // SoA with delta, rho, weight, nearestHigher, clusterIndex, layer, isSeed, and cellsCount fields in device global memory
  using HGCalSoARecHitsExtraDeviceCollection = PortableCollection<HGCalSoARecHitsExtra>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_PortableTestObjects_interface_alpaka_HGCalSoARecHitsExtraDeviceCollection_h
