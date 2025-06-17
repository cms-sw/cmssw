#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelMorphingConfig_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelMorphingConfig_h

#include <cstdint>
#include <vector>

struct SiPixelMorphingConfig {
  std::array<int32_t, 9> kernel1_;
  std::array<int32_t, 9> kernel2_;
};

#endif
