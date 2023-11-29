#ifndef DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersCollection_h
#define DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersCollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  using SiPixelClustersCollection = SiPixelClustersHost;
#else
  using SiPixelClustersCollection = SiPixelClustersDevice<Device>;
#endif
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace cms::alpakatools {
  template <>
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersCollection> {
    template <typename TQueue>
    static auto copyAsync(TQueue &queue, ALPAKA_ACCELERATOR_NAMESPACE::SiPixelClustersCollection const &srcData) {
      SiPixelClustersHost dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      dstData.setNClusters(srcData.nClusters(), srcData.offsetBPIX2());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_SiPixelClusterSoA_interface_alpaka_SiPixelClustersCollection_h
