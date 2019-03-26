#ifndef DataFormats_SiPixelDigi_PixelErrors_h
#define DataFormats_SiPixelDigi_PixelErrors_h

#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "FWCore/Utilities/interface/typedefs.h"

#include <map>
#include <vector>

// Better ideas for the placement of these?

struct PixelErrorCompact {
  uint32_t rawId;
  uint32_t word;
  unsigned char errorType;
  unsigned char fedId;
};

using PixelFormatterErrors = std::map<cms_uint32_t, std::vector<SiPixelRawDataError>>;

#endif
