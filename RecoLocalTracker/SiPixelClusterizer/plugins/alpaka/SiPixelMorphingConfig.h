#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h

#include <cstdint>

struct SiPixelMorphingConfig {
  bool applyDigiMorphing;
  uint32_t maxFakesInModule;
  uint32_t numMorphingModules;
};

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h
