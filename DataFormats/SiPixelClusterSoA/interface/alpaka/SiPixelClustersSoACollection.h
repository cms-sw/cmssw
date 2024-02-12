#ifndef DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersSoACollection_h
#define DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersSoACollection_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using SiPixelClustersSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, SiPixelClustersHost, SiPixelClustersDevice<Device>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<SiPixelClustersDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, SiPixelClustersDevice<TDevice> const &srcData) {
      // SiPixelClustersHost and SiPixelClustersDevice have a capacity larger than the ctor argument by one
      SiPixelClustersHost dstData(srcData->metadata().size() - 1, queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      dstData.setNClusters(srcData.nClusters(), srcData.offsetBPIX2());
#ifdef GPU_DEBUG  //keeping this untiil copies are in the Tracer
      printf("SiPixelClustersSoACollection: I'm copying to host.\n");
#endif
      return dstData;
    }
  };
}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(SiPixelClustersSoACollection, SiPixelClustersHost);
#endif  // DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersSoACollection_h
