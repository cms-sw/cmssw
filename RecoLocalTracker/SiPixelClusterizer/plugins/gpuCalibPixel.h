#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <limits>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainForHLTonGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

// local include(s)
#include "SiPixelClusterThresholds.h"

namespace gpuCalibPixel {

  using gpuClustering::invalidModuleId;

  // template <bool isRun2>
  __global__ void calibDigis(SiPixelClusterThresholds clusterThresholds,
                             uint16_t* id,
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

    const float VCaltoElectronGain = clusterThresholds.vCaltoElectronGain;
    const float VCaltoElectronGain_L1 = clusterThresholds.vCaltoElectronGain_L1;
    const float VCaltoElectronOffset = clusterThresholds.vCaltoElectronOffset;
    const float VCaltoElectronOffset_L1 = clusterThresholds.vCaltoElectronOffset_L1;

    // zero for next kernels...
    if (0 == first)
      clusModuleStart[0] = moduleStart[0] = 0;
    for (int i = first; i < phase1PixelTopology::numberOfModules; i += gridDim.x * blockDim.x) {
      nClustersInModule[i] = 0;
    }

    for (int i = first; i < numElements; i += gridDim.x * blockDim.x) {
      if (invalidModuleId == id[i])
        continue;

      bool isDeadColumn = false, isNoisyColumn = false;

      int row = x[i];
      int col = y[i];
      auto ret = ped->getPedAndGain(id[i], col, row, isDeadColumn, isNoisyColumn);
      float pedestal = ret.first;
      float gain = ret.second;
      // float pedestal = 0; float gain = 1.;
      if (isDeadColumn | isNoisyColumn) {
        printf("bad pixel at %d in %d\n", i, id[i]);
        id[i] = invalidModuleId;
        adc[i] = 0;
      } else {
        float vcal = float(adc[i]) * gain - pedestal * gain;

        float conversionFactor = id[i] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain;
        float offset = id[i] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset;
        vcal = vcal * conversionFactor + offset;

        adc[i] = std::clamp(int(vcal), 100, int(std::numeric_limits<uint16_t>::max()));
      }
    }
  }

  __global__ void calibDigisPhase2(SiPixelClusterThresholds clusterThresholds,
                                   uint16_t* id,
                                   uint16_t* adc,
                                   int numElements,
                                   uint32_t* __restrict__ moduleStart,        // just to zero first
                                   uint32_t* __restrict__ nClustersInModule,  // just to zero them
                                   uint32_t* __restrict__ clusModuleStart     // just to zero first
  ) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    // zero for next kernels...

    const float ElectronPerADCGain = clusterThresholds.electronPerADCGain;
    const int8_t Phase2ReadoutMode = clusterThresholds.phase2ReadoutMode;
    const uint16_t Phase2DigiBaseline = clusterThresholds.phase2DigiBaseline;
    const uint8_t Phase2KinkADC = clusterThresholds.phase2KinkADC;

    if (0 == first)
      clusModuleStart[0] = moduleStart[0] = 0;
    for (int i = first; i < phase2PixelTopology::numberOfModules; i += gridDim.x * blockDim.x) {
      nClustersInModule[i] = 0;
    }

    for (int i = first; i < numElements; i += gridDim.x * blockDim.x) {
      if (invalidModuleId == id[i])
        continue;

      const int mode = (Phase2ReadoutMode < -1 ? -1 : Phase2ReadoutMode);

      int adc_int = adc[i];

      if (mode < 0)
        adc_int = int(adc_int * ElectronPerADCGain);
      else {
        if (adc_int < Phase2KinkADC)
          adc_int = int((adc_int + 0.5) * ElectronPerADCGain);
        else {
          const int8_t dspp = (Phase2ReadoutMode < 10 ? Phase2ReadoutMode : 10);
          const int8_t ds = int8_t(dspp <= 1 ? 1 : (dspp - 1) * (dspp - 1));

          adc_int -= Phase2KinkADC;
          adc_int *= ds;
          adc_int += Phase2KinkADC;

          adc_int = ((adc_int + 0.5 * ds) * ElectronPerADCGain);
        }

        adc_int += int(Phase2DigiBaseline);
      }
      adc[i] = std::min(adc_int, int(std::numeric_limits<uint16_t>::max()));
    }
  }

}  // namespace gpuCalibPixel

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
