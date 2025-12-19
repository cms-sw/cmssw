
#pragma once

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/ClusterMaskDevice.h"

#include <alpaka/alpaka.hpp>
#include <cstdint>

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  struct KernelFilterLayerClusterByAlgoAndSize {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                  std::span<const int32_t> algoNumber,
                                  ticl::ClusterMaskDevice::View layerClusterMask,
                                  uint32_t minClusterSize,
                                  uint32_t maxClusterSize) const;
  };

  class ClusterFilterByAlgoAndSize {
    void filter(Queue& queue,
                const reco::CaloClusterDeviceCollection& layerClusters,
                uint32_t minClusterSize,
                uint32_t maxClusterSize);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl
