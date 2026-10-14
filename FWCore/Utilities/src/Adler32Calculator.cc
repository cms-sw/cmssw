#include "FWCore/Utilities/interface/Adler32Calculator.h"

#include <zlib.h>

namespace cms {

  uint32_t adler32(char const* data, size_t len, uint32_t adler) {
    return adler32_z(adler, reinterpret_cast<Bytef const*>(data), len);
  }

}  // namespace cms
