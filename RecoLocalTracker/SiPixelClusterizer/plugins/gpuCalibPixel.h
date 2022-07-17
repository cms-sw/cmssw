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

namespace gpuCalibPixel {

  using gpuClustering::invalidModuleId;

  // calibrationConstants
  // valid for run2
  constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
  constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
  constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
  constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220
  constexpr int VCalChargeThreshold = 100;
  //for phase2
  constexpr float ElectronPerADCGain = 600;
  constexpr int8_t Phase2ReadoutMode = 3;
  constexpr uint16_t Phase2DigiBaseline = 1500;
  constexpr uint8_t Phase2KinkADC = 8;

  template <bool isRun2>
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
        if constexpr (isRun2) {
          float conversionFactor = id[i] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain;
          float offset = id[i] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset;
          vcal = vcal * conversionFactor + offset;
        }
        adc[i] = std::clamp(int(vcal), 100, int(std::numeric_limits<uint16_t>::max()));
      }
    }
  }

  __global__ void calibDigisPhase2(uint16_t* id,
                                   uint16_t* adc,
                                   int numElements,
                                   uint32_t* __restrict__ moduleStart,        // just to zero first
                                   uint32_t* __restrict__ nClustersInModule,  // just to zero them
                                   uint32_t* __restrict__ clusModuleStart     // just to zero first
  ) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    // zero for next kernels...

    if (0 == first)
      clusModuleStart[0] = moduleStart[0] = 0;
    for (int i = first; i < phase2PixelTopology::numberOfModules; i += gridDim.x * blockDim.x) {
      nClustersInModule[i] = 0;
    }

    for (int i = first; i < numElements; i += gridDim.x * blockDim.x) {
      if (invalidModuleId == id[i])
        continue;

      constexpr int mode = (Phase2ReadoutMode < -1 ? -1 : Phase2ReadoutMode);

      int adc_int = adc[i];

      if constexpr (mode < 0)
        adc_int = int(adc_int * ElectronPerADCGain);
      else {
        if (adc_int < Phase2KinkADC)
          adc_int = int((adc_int - 0.5) * ElectronPerADCGain);
        else {
          constexpr int8_t dspp = (Phase2ReadoutMode < 10 ? Phase2ReadoutMode : 10);
          constexpr int8_t ds = int8_t(dspp <= 1 ? 1 : (dspp - 1) * (dspp - 1));

          adc_int -= (Phase2KinkADC - 1);
          adc_int *= ds;
          adc_int += (Phase2KinkADC - 1);

          adc_int = ((adc_int + 0.5 * ds) * ElectronPerADCGain);
        }

        adc_int += int(Phase2DigiBaseline);
      }

      adc[i] = std::min(adc_int, int(std::numeric_limits<uint16_t>::max()));
    }
  }

}  // namespace gpuCalibPixel

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
