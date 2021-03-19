#ifndef DataFormats_SiPixelRawData_interface_SiPixelErrorCompact_h
#define DataFormats_SiPixelRawData_interface_SiPixelErrorCompact_h

#include <cstdint>

struct SiPixelErrorCompact {
  uint32_t rawId;
  uint32_t word;
  uint8_t errorType;
  uint8_t fedId;
};

#endif  // DataFormats_SiPixelRawData_interface_SiPixelErrorCompact_h
