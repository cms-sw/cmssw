#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_CalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_CalibPixel_h

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTLayout.h"
#include "CondFormats/SiPixelObjects/interface/alpaka/SiPixelGainCalibrationForHLTDevice.h"
#include "CondFormats/SiPixelObjects/interface/alpaka/SiPixelGainCalibrationForHLTUtilities.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterThresholds.h"

//#define GPU_DEBUG

namespace calibPixel {
  using namespace cms::alpakatools;

  constexpr uint16_t InvId = std::numeric_limits<uint16_t>::max() - 1;
  // must be > MaxNumModules

  struct CalibDigis {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  SiPixelClusterThresholds clusterThresholds,
                                  SiPixelDigisSoAView view,
                                  SiPixelClustersSoAView clus_view,
                                  const SiPixelGainCalibrationForHLTSoAConstView gains,
                                  int numElements) const {
      const float VCaltoElectronGain = clusterThresholds.vCaltoElectronGain;
      const float VCaltoElectronGain_L1 = clusterThresholds.vCaltoElectronGain_L1;
      const float VCaltoElectronOffset = clusterThresholds.vCaltoElectronOffset;
      const float VCaltoElectronOffset_L1 = clusterThresholds.vCaltoElectronOffset_L1;

      // zero for next kernels...
      if (cms::alpakatools::once_per_grid(acc)) {
        clus_view[0].clusModuleStart() = clus_view[0].moduleStart() = 0;
      }

      cms::alpakatools::for_each_element_in_grid_strided(
          acc, phase1PixelTopology::numberOfModules, [&](uint32_t i) { clus_view[i].clusInModule() = 0; });
      cms::alpakatools::for_each_element_in_grid_strided(acc, numElements, [&](uint32_t i) {
        auto dvgi = view[i];
        if (dvgi.moduleId() != InvId) {
          bool isDeadColumn = false, isNoisyColumn = false;
          int row = dvgi.xx();
          int col = dvgi.yy();
          auto ret = SiPixelGainUtilities::getPedAndGain(gains, dvgi.moduleId(), col, row, isDeadColumn, isNoisyColumn);
          float pedestal = ret.first;
          float gain = ret.second;
          if (isDeadColumn | isNoisyColumn) {
            dvgi.moduleId() = InvId;
            dvgi.adc() = 0;
            printf("bad pixel at %d in %d\n", i, dvgi.moduleId());
          } else {
            float vcal = dvgi.adc() * gain - pedestal * gain;

            float conversionFactor = dvgi.moduleId() < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain;
            float offset = dvgi.moduleId() < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset;
#ifdef GPU_DEBUG
            auto old_adc = dvgi.adc();
#endif
            dvgi.adc() = std::max(100, int(vcal * conversionFactor + offset));
#ifdef GPU_DEBUG
            if (cms::alpakatools::once_per_grid(acc)) {
              printf(
                  "module %d pixel %d -> old_adc = %d; vcal = %.2f; conversionFactor = %.2f; offset = %.2f; new_adc = "
                  "%d \n",
                  dvgi.moduleId(),
                  i,
                  old_adc,
                  vcal,
                  conversionFactor,
                  offset,
                  dvgi.adc());
            }
#endif
          }
        }
      });
    }
  };
  struct CalibDigisPhase2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  SiPixelClusterThresholds clusterThresholds,
                                  SiPixelDigisSoAView view,
                                  SiPixelClustersSoAView clus_view,
                                  int numElements) const {
      const float ElectronPerADCGain = clusterThresholds.electronPerADCGain;
      const int8_t Phase2ReadoutMode = clusterThresholds.phase2ReadoutMode;
      const uint16_t Phase2DigiBaseline = clusterThresholds.phase2DigiBaseline;
      const uint8_t Phase2KinkADC = clusterThresholds.phase2KinkADC;

      // zero for next kernels...
      if (cms::alpakatools::once_per_grid(acc)) {
        clus_view[0].clusModuleStart() = clus_view[0].moduleStart() = 0;
      }

      cms::alpakatools::for_each_element_in_grid_strided(
          acc, phase2PixelTopology::numberOfModules, [&](uint32_t i) { clus_view[i].clusInModule() = 0; });
      cms::alpakatools::for_each_element_in_grid_strided(acc, numElements, [&](uint32_t i) {
        auto dvgi = view[i];
        if (pixelClustering::invalidModuleId != dvgi.moduleId()) {
          const int mode = (Phase2ReadoutMode < -1 ? -1 : Phase2ReadoutMode);
          int adc_int = dvgi.adc();
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
          dvgi.adc() = std::min(adc_int, int(std::numeric_limits<uint16_t>::max()));
        }
      });
    }
  };
}  // namespace calibPixel

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_CalibPixel_h
