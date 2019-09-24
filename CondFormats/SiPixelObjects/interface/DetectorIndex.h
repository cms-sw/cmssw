#include <cstdint>
#ifndef CondFormats_SiPixelObjects_DetectorIndex_H
#define CondFormats_SiPixelObjects_DetectorIndex_H

namespace sipixelobjects {
  struct DetectorIndex {
    uint32_t rawId;
    int row;
    int col;
  };
}  // namespace sipixelobjects
#endif
