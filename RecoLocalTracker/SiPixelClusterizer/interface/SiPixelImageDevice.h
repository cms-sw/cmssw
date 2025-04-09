#ifndef RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageDevice_h
#define RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageHost.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using SiPixelImageDevice = PortableCollection<SiPixelImageSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(SiPixelImageDevice, SiPixelImageHost)

#endif  // RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageDevice_h
