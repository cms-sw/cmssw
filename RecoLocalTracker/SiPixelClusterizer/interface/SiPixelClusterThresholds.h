#ifndef RecoLocalTracker_SiPixelClusterizer_interface_SiPixelClusterThresholds_h
#define RecoLocalTracker_SiPixelClusterizer_interface_SiPixelClusterThresholds_h

/* This struct is an implementation detail of this package.
 * It's in the interface directory because it needs to be shared by the legacy, CUDA, and Alpaka plugins.
 */

struct SiPixelClusterThresholds {
  inline constexpr int32_t getThresholdForLayerOnCondition(bool isLayer1) const noexcept {
    return isLayer1 ? layer1 : otherLayers;
  }
  const int32_t layer1 = 0;
  const int32_t otherLayers = 0;

  const float vCaltoElectronGain = 0;
  const float vCaltoElectronGain_L1 = 0;
  const float vCaltoElectronOffset = 0;
  const float vCaltoElectronOffset_L1 = 0;

  const float electronPerADCGain = 0;
  const int8_t phase2ReadoutMode = 0;
  const uint16_t phase2DigiBaseline = 0;
  const uint8_t phase2KinkADC = 0;

  //Basic just for thresholds
  SiPixelClusterThresholds(const int32_t llayer1, const int32_t lotherLayers)
      : layer1(llayer1), otherLayers(lotherLayers) {}

  //For Phase1
  SiPixelClusterThresholds(const int32_t llayer1,
                           const int32_t lotherLayers,
                           const float lvCaltoElectronGain,
                           const float lvCaltoElectronGain_L1,
                           const float lvCaltoElectronOffset,
                           const float lvCaltoElectronOffset_L1)
      : layer1(llayer1),
        otherLayers(lotherLayers),
        vCaltoElectronGain(lvCaltoElectronGain),
        vCaltoElectronGain_L1(lvCaltoElectronGain_L1),
        vCaltoElectronOffset(lvCaltoElectronOffset),
        vCaltoElectronOffset_L1(lvCaltoElectronOffset_L1) {}

  //For Phase2
  SiPixelClusterThresholds(const int32_t llayer1,
                           const int32_t lotherLayers,
                           const float lelectronPerADCGain,
                           const int8_t lphase2ReadoutMode,
                           const uint16_t lphase2DigiBaseline,
                           const uint8_t lphase2KinkADC)
      : layer1(llayer1),
        otherLayers(lotherLayers),
        electronPerADCGain(lelectronPerADCGain),
        phase2ReadoutMode(lphase2ReadoutMode),
        phase2DigiBaseline(lphase2DigiBaseline),
        phase2KinkADC(lphase2KinkADC) {}
};

#endif  // RecoLocalTracker_SiPixelClusterizer_interface_SiPixelClusterThresholds_h
