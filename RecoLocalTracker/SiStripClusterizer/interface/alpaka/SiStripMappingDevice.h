#ifndef RecoLocalTracker_SiStripClusterizer_interface_alpaka_SiStripMappingDevice_h
#define RecoLocalTracker_SiStripClusterizer_interface_alpaka_SiStripMappingDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripMappingHost.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripMappingSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace ::sistrip;
  using SiStripMappingDevice = PortableCollection<SiStripMappingSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

// check that the sistrip device collection for the host device is the same as the sistrip host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(sistrip::SiStripMappingDevice, sistrip::SiStripMappingHost);

#endif  // RecoLocalTracker_SiStripClusterizer_interface_alpaka_SiStripMappingDevice_h
