
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/ClusterMaskDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoHGCal/TICL/plugins/alpaka/ClusterFilterByAlgoAndSize.h"

#include <alpaka/alpaka.hpp>
#include <cstdint>

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  template <typename TAcc>
  ALPAKA_FN_ACC void KernelFilterLayerClusterByAlgoAndSize::operator()(
      const TAcc& acc,
      reco::CaloClusterDeviceCollection::ConstView layerClusters,
      std::span<const int32_t> algoNumber,
      ticl::ClusterMaskDevice::View layerClusterMask,
      uint32_t minClusterSize,
      uint32_t maxClusterSize) const {
    for (auto lcIdx : alpaka::uniformElements(acc, layerClusters.position().metadata().size())) {
      bool foundAlgoNumber = false;
      for (auto algo : algoNumber) {
        if (algo == *layerClusters[lcIdx].algoID())
          foundAlgoNumber = true;
      }
      const auto layerClusterSize = hitsAssociations[lcIdx].size();
      if (foundAlgoNumber && (layerClusterSize < minClusterSize) || (layerClusterSize > maxClusterSize))
        layerClusterMask[lcIdx] = 0.f;
      else
        layerClusterMask[lcIdx] = 1.f;
    }
  }

  void ClusterFilterByAlgoAndSize::filter(Queue& queue,
                                          const reco::CaloClusterDeviceCollection& layerClusters,
                                          uint32_t minClusterSize,
                                          uint32_t maxClusterSize) {
    auto blocksize = 1024;
    auto gridsize = cms::alpakatools::divide_up_by(layerClusters.metadata().size(), blocksize);
    auto workdivision = cms::alpakatools::make_workdiv<Acc1D>(gridsize, blocksize);
    alpaka::exec<Acc1D>(
        queue, workdivision, KernelFilterLayerClusterByAlgoAndSize{}, layerClusters.view(), hitsAssociations.view());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl
