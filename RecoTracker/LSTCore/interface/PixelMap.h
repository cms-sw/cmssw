#ifndef RecoTracker_LSTCore_interface_PixelMap_h
#define RecoTracker_LSTCore_interface_PixelMap_h

#include <vector>
#include <cstdint>

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  struct PixelMap {
    uint16_t pixelModuleIndex;

    std::vector<unsigned int> connectedPixelsIndex;
    std::vector<unsigned int> connectedPixelsSizes;
    std::vector<unsigned int> connectedPixelsIndexPos;
    std::vector<unsigned int> connectedPixelsSizesPos;
    std::vector<unsigned int> connectedPixelsIndexNeg;
    std::vector<unsigned int> connectedPixelsSizesNeg;

    PixelMap(unsigned int sizef = size_superbins)
        : pixelModuleIndex(0),
          connectedPixelsIndex(sizef),
          connectedPixelsSizes(sizef),
          connectedPixelsIndexPos(sizef),
          connectedPixelsSizesPos(sizef),
          connectedPixelsIndexNeg(sizef),
          connectedPixelsSizesNeg(sizef) {}
  };
}  // namespace lst

#endif
