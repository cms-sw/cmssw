
#pragma once

#include "DataFormats/TICL/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/ClusterMaskDevice.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <alpaka/alpaka.hpp>
#include <cstdint>

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  struct KernelFilterLayerClusterByAlgoAndSize {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                  std::span<int32_t> algoNumber,
                                  ticl::ClusterMaskDevice::View layerClusterMask,
                                  uint32_t minClusterSize,
                                  uint32_t maxClusterSize) const;
  };

  class ClusterFilterByAlgoAndSize {
  public:
    ClusterFilterByAlgoAndSize(const edm::ParameterSet& config)
        : algo_number_{config.getParameter<int32_t>("algo_number")} {}
    void filter(Queue& queue,
                const reco::CaloClusterDeviceCollection& layerClusters,
                ticl::ClusterMaskDevice& layerClusterMask,
                uint32_t minClusterSize,
                uint32_t maxClusterSize);

  private:
    std::vector<int32_t> algo_number_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl
