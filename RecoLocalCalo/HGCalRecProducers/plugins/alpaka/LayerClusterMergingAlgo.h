
#pragma once

#include "DataFormats/TICL/interface/alpaka/CaloClusterDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct KernelMergeLayerClusters {
    ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                  reco::CaloClusterDeviceCollection::View merged,
                                  reco::CaloClusterDeviceCollection::ConstView input,
                                  uint32_t start) const;
  };

  struct LayerClusterMerger {
    void merge(Queue& queue,
               reco::CaloClusterDeviceCollection::View merged,
               reco::CaloClusterDeviceCollection::ConstView input,
               uint32_t& start);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
