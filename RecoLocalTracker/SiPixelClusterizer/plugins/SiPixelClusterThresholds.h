#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterThresholds_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterThresholds_h

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
  SiPixelClusterThresholds(const int32_t layer1, const int32_t otherLayers)
      : layer1(layer1), otherLayers(otherLayers) {}

  //For Phase1
  SiPixelClusterThresholds(const int32_t layer1,
                           const int32_t otherLayers,
                           const float vCaltoElectronGain,
                           const float vCaltoElectronGain_L1,
                           const float vCaltoElectronOffset,
                           const float vCaltoElectronOffset_L1)
      : layer1(layer1),
        otherLayers(otherLayers),
        vCaltoElectronGain(vCaltoElectronGain),
        vCaltoElectronGain_L1(vCaltoElectronGain_L1),
        vCaltoElectronOffset(vCaltoElectronOffset),
        vCaltoElectronOffset_L1(vCaltoElectronOffset_L1) {}

  //For Phase2
  SiPixelClusterThresholds(const int32_t layer1,
                           const int32_t otherLayers,
                           const float electronPerADCGain,
                           const int8_t phase2ReadoutMode,
                           const uint16_t phase2DigiBaseline,
                           const uint8_t phase2KinkADC)
      : layer1(layer1),
        otherLayers(otherLayers),
        electronPerADCGain(electronPerADCGain),
        phase2ReadoutMode(phase2ReadoutMode),
        phase2DigiBaseline(phase2DigiBaseline),
        phase2KinkADC(phase2KinkADC) {}
};

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterThresholds_h
