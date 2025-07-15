#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h

#include <cstdint>
#include <vector>

struct SiPixelMorphingConfig {
  std::array<int32_t, 9> kernel1;
  std::array<int32_t, 9> kernel2;
};

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_alpaka_SiPixelMorphingConfig_h
