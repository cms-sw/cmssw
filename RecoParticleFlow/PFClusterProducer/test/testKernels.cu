#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "testKernels.h"

namespace testPFlowCUDA {

  __global__ void printPFClusteringParamsOnGPU(PFClusteringParamsGPU::DeviceProduct::ConstView params) {
    printf("nNeigh: %d\n", params.nNeigh());
    printf("seedPt2ThresholdEB: %4.2f\n", params.seedPt2ThresholdEB());
    printf("seedPt2ThresholdEE: %4.2f\n", params.seedPt2ThresholdEE());

    for (uint32_t idx = 0; idx < 4; ++idx)
      printf("seedEThresholdEB_vec[%d]: %4.2f\n", idx, params.seedEThresholdEB_vec()[idx]);
    for (uint32_t idx = 0; idx < 7; ++idx)
      printf("seedEThresholdEE_vec[%d]: %4.2f\n", idx, params.seedEThresholdEE_vec()[idx]);

    for (uint32_t idx = 0; idx < 4; ++idx)
      printf("topoEThresholdEB_vec[%d]: %4.2f\n", idx, params.topoEThresholdEB_vec()[idx]);
    for (uint32_t idx = 0; idx < 7; ++idx)
      printf("topoEThresholdEE_vec[%d]: %4.2f\n", idx, params.topoEThresholdEE_vec()[idx]);

    printf("showerSigma2: %4.2f\n", params.showerSigma2());
    printf("minFracToKeep: %4.2f\n", params.minFracToKeep());
    printf("minFracTot: %4.2f\n", params.minFracTot());
    printf("maxIterations: %d\n", params.maxIterations());
    printf("excludeOtherSeeds: %d\n", params.excludeOtherSeeds());
    printf("stoppingTolerance: %4.2f\n", params.stoppingTolerance());
    printf("minFracInCalc: %4.2f\n", params.minFracInCalc());
    printf("minAllowedNormalization: %4.2f\n", params.minAllowedNormalization());

    for (uint32_t idx = 0; idx < 4; ++idx)
      printf("recHitEnergyNormInvEB_vec[%d]: %4.2f\n", idx, params.recHitEnergyNormInvEB_vec()[idx]);
    for (uint32_t idx = 0; idx < 7; ++idx)
      printf("recHitEnergyNormInvEE_vec[%d]: %4.2f\n", idx, params.recHitEnergyNormInvEE_vec()[idx]);

    printf("barrelTimeResConsts_corrTermLowE: %4.2f\n", params.barrelTimeResConsts_corrTermLowE());
    printf("barrelTimeResConsts_threshLowE: %4.2f\n", params.barrelTimeResConsts_threshLowE());
    printf("barrelTimeResConsts_noiseTerm: %4.2f\n", params.barrelTimeResConsts_noiseTerm());
    printf("barrelTimeResConsts_constantTermLowE2: %4.2f\n", params.barrelTimeResConsts_constantTermLowE2());
    printf("barrelTimeResConsts_noiseTermLowE: %4.2f\n", params.barrelTimeResConsts_noiseTermLowE());
    printf("barrelTimeResConsts_threshHighE: %4.2f\n", params.barrelTimeResConsts_threshHighE());
    printf("barrelTimeResConsts_constantTerm2: %4.2f\n", params.barrelTimeResConsts_constantTerm2());
    printf("barrelTimeResConsts_resHighE2: %4.2f\n", params.barrelTimeResConsts_resHighE2());

    printf("endcapTimeResConsts_corrTermLowE: %4.2f\n", params.endcapTimeResConsts_corrTermLowE());
    printf("endcapTimeResConsts_threshLowE: %4.2f\n", params.endcapTimeResConsts_threshLowE());
    printf("endcapTimeResConsts_noiseTerm: %4.2f\n", params.endcapTimeResConsts_noiseTerm());
    printf("endcapTimeResConsts_constantTermLowE2: %4.2f\n", params.endcapTimeResConsts_constantTermLowE2());
    printf("endcapTimeResConsts_noiseTermLowE: %4.2f\n", params.endcapTimeResConsts_noiseTermLowE());
    printf("endcapTimeResConsts_threshHighE: %4.2f\n", params.endcapTimeResConsts_threshHighE());
    printf("endcapTimeResConsts_constantTerm2: %4.2f\n", params.endcapTimeResConsts_constantTerm2());
    printf("endcapTimeResConsts_resHighE2: %4.2f\n", params.endcapTimeResConsts_resHighE2());
  }

}  // namespace testPFlowCUDA

namespace testPFlow {

  void testPFClusteringParamsEntryPoint(PFClusteringParamsGPU::DeviceProduct const& pfClusParams,
                                        cudaStream_t cudaStream) {
    testPFlowCUDA::printPFClusteringParamsOnGPU<<<1, 1, 0, cudaStream>>>(pfClusParams.const_view());
  }

}  // namespace testPFlow
