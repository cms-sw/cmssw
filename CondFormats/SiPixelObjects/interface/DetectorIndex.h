#ifndef CondFormats_SiPixelObjects_DetectorIndex_H
#define CondFormats_SiPixelObjects_DetectorIndex_H

#include <boost/cstdint.hpp>

namespace sipixelobjects {
  struct DetectorIndex { uint32_t rawId; int row; int col; };
}
#endif
