// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesConstants.h"

#include "HGCalLayerClustersAlgoWrapper.h"
#include "ConstantsForClusters.h"

#include "CLUEAlgoAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using namespace hgcal::constants;

  void HGCalLayerClustersAlgoWrapper::run(Queue& queue,
                                          const unsigned int size,
                                          const float dc,
                                          const float kappa,
                                          const float outlierDeltaFactor,
                                          const HGCalSoARecHitsDeviceCollection::ConstView inputs,
                                          HGCalSoARecHitsExtraDeviceCollection::View outputs) const {
    CLUEAlgoAlpaka<ALPAKA_ACCELERATOR_NAMESPACE::Acc1D, Queue, HGCalSiliconTilesConstants, kHGCalLayers> algoStandalone(
        queue, dc, kappa, outlierDeltaFactor, false);

    // Initialize output memory to 0
    auto delta = cms::alpakatools::make_device_view<float>(queue, outputs.delta(), size);
    alpaka::memset(queue, delta, 0x0);
    auto rho = cms::alpakatools::make_device_view<float>(queue, outputs.rho(), size);
    alpaka::memset(queue, rho, 0x0);
    auto nearestHigher = cms::alpakatools::make_device_view<unsigned int>(queue, outputs.nearestHigher(), size);
    alpaka::memset(queue, nearestHigher, 0x0);
    auto clusterIndex = cms::alpakatools::make_device_view<int>(queue, outputs.clusterIndex(), size);
    alpaka::memset(queue, clusterIndex, kInvalidClusterByte);
    auto isSeed = cms::alpakatools::make_device_view<uint8_t>(queue, outputs.isSeed(), size);
    alpaka::memset(queue, isSeed, 0x0);

    algoStandalone.makeClustersCMSSW(size,
                                     inputs.dim1(),
                                     inputs.dim2(),
                                     inputs.layer(),
                                     inputs.weight(),
                                     inputs.sigmaNoise(),
                                     inputs.detid(),
                                     outputs.rho(),
                                     outputs.delta(),
                                     outputs.nearestHigher(),
                                     outputs.clusterIndex(),
                                     outputs.isSeed(),
                                     &outputs.numberOfClustersScalar());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
