#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h

#include <cstdint>
#include <cstdio>

#include "CondFormats/SiPixelObjects/interface/SiPixelGainForHLTonGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "gpuClusteringConstants.h"

namespace gpuCalibPixel {

  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

  constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
  constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
  constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
  constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220

  __global__ void calibDigis(uint16_t* id,
                             uint16_t const* __restrict__ x,
                             uint16_t const* __restrict__ y,
                             uint16_t* adc,
                             SiPixelGainForHLTonGPU const* __restrict__ ped,
                             int numElements,
                             uint32_t* __restrict__ moduleStart,        // just to zero first
                             uint32_t* __restrict__ nClustersInModule,  // just to zero them
                             uint32_t* __restrict__ clusModuleStart     // just to zero first
  ) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;

    // zero for next kernels...
    if (0 == first)
      clusModuleStart[0] = moduleStart[0] = 0;
    for (int i = first; i < gpuClustering::MaxNumModules; i += gridDim.x * blockDim.x) {
      nClustersInModule[i] = 0;
    }

    for (int i = first; i < numElements; i += gridDim.x * blockDim.x) {
      if (InvId == id[i])
        continue;

      float conversionFactor = id[i] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain;
      float offset = id[i] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset;

      bool isDeadColumn = false, isNoisyColumn = false;

      int row = x[i];
      int col = y[i];
      auto ret = ped->getPedAndGain(id[i], col, row, isDeadColumn, isNoisyColumn);
      float pedestal = ret.first;
      float gain = ret.second;
      // float pedestal = 0; float gain = 1.;
      if (isDeadColumn | isNoisyColumn) {
        id[i] = InvId;
        adc[i] = 0;
        printf("bad pixel at %d in %d\n", i, id[i]);
      } else {
        float vcal = adc[i] * gain - pedestal * gain;
        adc[i] = std::max(100, int(vcal * conversionFactor + offset));
      }
    }
  }
}  // namespace gpuCalibPixel

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
