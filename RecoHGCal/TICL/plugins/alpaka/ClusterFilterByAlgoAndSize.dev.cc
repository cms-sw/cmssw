
#include "DataFormats/TICL/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/ClusterMaskDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoHGCal/TICL/plugins/alpaka/ClusterFilterByAlgoAndSize.h"

#include <alpaka/alpaka.hpp>
#include <cstdint>
#include <span>

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  template <typename TAcc>
  ALPAKA_FN_ACC void KernelFilterLayerClusterByAlgoAndSize::operator()(
      const TAcc& acc,
      reco::CaloClusterDeviceCollection::ConstView layerClusters,
      std::span<int32_t> algoNumber,
      ticl::ClusterMaskDevice::View layerClusterMask,
      uint32_t minClusterSize,
      uint32_t maxClusterSize) const {
    for (auto lcIdx : alpaka::uniformElements(acc, layerClusters.position().metadata().size())) {
      bool foundAlgoNumber = false;
      for (auto algo : algoNumber) {
        if (algo == layerClusters.indexes()[lcIdx].algoID())
          foundAlgoNumber = true;
      }
      const auto layerClusterSize = static_cast<uint32_t>(layerClusters.position()[lcIdx].cells());
      if (foundAlgoNumber && (layerClusterSize < minClusterSize) || (layerClusterSize > maxClusterSize))
        layerClusterMask[lcIdx] = 0.f;
    }
  }

  void ClusterFilterByAlgoAndSize::filter(Queue& queue,
                                          const reco::CaloClusterDeviceCollection& layerClusters,
                                          ticl::ClusterMaskDevice& layerClusterMask,
                                          uint32_t minClusterSize,
                                          uint32_t maxClusterSize) {
    auto d_algo_number = cms::alpakatools::make_device_buffer<int32_t[]>(queue, algo_number_.size());
    alpaka::memcpy(queue, d_algo_number, cms::alpakatools::make_host_view(algo_number_.data(), algo_number_.size()));
    auto blocksize = 1024;
    auto gridsize = cms::alpakatools::divide_up_by(layerClusters.view().position().metadata().size(), blocksize);
    auto workdivision = cms::alpakatools::make_workdiv<Acc1D>(gridsize, blocksize);
    alpaka::exec<Acc1D>(queue,
                        workdivision,
                        KernelFilterLayerClusterByAlgoAndSize{},
                        layerClusters.view(),
                        std::span{d_algo_number.data(), algo_number_.size()},
                        layerClusterMask.view(),
                        minClusterSize,
                        maxClusterSize);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl
