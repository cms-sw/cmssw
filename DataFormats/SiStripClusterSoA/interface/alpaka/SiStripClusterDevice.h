#ifndef DataFormats_SiStripClusterSoA_interface_alpaka_SiStripClustersDevice_h
#define DataFormats_SiStripClusterSoA_interface_alpaka_SiStripClustersDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiStripClusterSoA/interface/SiStripClusterHost.h"
#include "DataFormats/SiStripClusterSoA/interface/SiStripClusterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  // make the names from the top-level sistrip namespace visible for unqualified lookup
  // inside the ALPAKA_ACCELERATOR_NAMESPACE::sistrip namespace
  using namespace ::sistrip;
  using SiStripClusterDevice = PortableCollection<SiStripClusterSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

// check that the sistrip device collection for the host device is the same as the sistrip host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(sistrip::SiStripClusterDevice, sistrip::SiStripClusterHost);

#endif  // DataFormats_SiStripClusterSoA_interface_alpaka_SiStripClustersDevice_h
