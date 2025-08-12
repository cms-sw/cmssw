#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h

#include <cstdint>
#include <vector>

struct SiPixelMorphingConfig {
  std::vector<uint32_t> morphingModules;
  bool applyDigiMorphing;
  uint32_t maxFakesInModule;
};

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h
