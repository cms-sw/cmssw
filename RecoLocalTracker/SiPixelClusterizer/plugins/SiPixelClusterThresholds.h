#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterThresholds_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterThresholds_h

struct SiPixelClusterThresholds {
  inline constexpr int32_t getThresholdForLayerOnCondition(bool isLayer1) const noexcept {
    return isLayer1 ? layer1 : otherLayers;
  }
  const int32_t layer1;
  const int32_t otherLayers;
};

constexpr SiPixelClusterThresholds kSiPixelClusterThresholdsDefaultPhase1{.layer1 = 2000, .otherLayers = 4000};
constexpr SiPixelClusterThresholds kSiPixelClusterThresholdsDefaultPhase2{.layer1 = 4000, .otherLayers = 4000};

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterThresholds_h
