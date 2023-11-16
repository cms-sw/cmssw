#ifndef DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersCollection_h
#define DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersCollection_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
using SiPixelClustersCollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, SiPixelClustersHost, SiPixelClustersDevice<Device>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<SiPixelClustersDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, SiPixelClustersDevice<TDevice> const &srcData) {
      SiPixelClustersHost dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      dstData.setNClusters(srcData.nClusters(), srcData.offsetBPIX2());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(SiPixelClustersCollection, SiPixelClustersHost);
#endif  // DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersCollection_h
