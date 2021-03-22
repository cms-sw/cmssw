#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterThresholds_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterThresholds_h

struct SiPixelClusterThresholds {
  inline constexpr int32_t getThresholdForLayerOnCondition(bool isLayer1) const noexcept {
    return isLayer1 ? layer1 : otherLayers;
  }
  const int32_t layer1;
  const int32_t otherLayers;
};

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelClusterThresholds_h
