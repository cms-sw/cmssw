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

    algoStandalone.makeClustersCMSSW(size,
                                     inputs.dim1(),
                                     inputs.dim2(),
                                     inputs.layer(),
                                     inputs.energy(),
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
