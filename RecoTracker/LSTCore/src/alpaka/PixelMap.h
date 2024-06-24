#ifndef PixelMap_h
#define PixelMap_h

#include <vector>
#include <cstdint>

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#else
#include "Constants.h"
#endif

namespace SDL {
  // PixelMap is never allocated on the device.
  // This is also not passed to any of the kernels, so we can combine the structs.
  struct pixelMap {
    uint16_t pixelModuleIndex;

    std::vector<unsigned int> connectedPixelsIndex;
    std::vector<unsigned int> connectedPixelsSizes;
    std::vector<unsigned int> connectedPixelsIndexPos;
    std::vector<unsigned int> connectedPixelsSizesPos;
    std::vector<unsigned int> connectedPixelsIndexNeg;
    std::vector<unsigned int> connectedPixelsSizesNeg;

    int* pixelType;

    pixelMap(unsigned int sizef = size_superbins)
        : pixelModuleIndex(0),
          connectedPixelsIndex(sizef),
          connectedPixelsSizes(sizef),
          connectedPixelsIndexPos(sizef),
          connectedPixelsSizesPos(sizef),
          connectedPixelsIndexNeg(sizef),
          connectedPixelsSizesNeg(sizef) {}
  };
}  // namespace SDL

#endif
