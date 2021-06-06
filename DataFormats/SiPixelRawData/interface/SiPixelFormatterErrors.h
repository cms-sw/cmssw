#ifndef DataFormats_SiPixelRawData_interface_SiPixelFormatterErrors_h
#define DataFormats_SiPixelRawData_interface_SiPixelFormatterErrors_h

#include <map>
#include <vector>

#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "FWCore/Utilities/interface/typedefs.h"

using SiPixelFormatterErrors = std::map<cms_uint32_t, std::vector<SiPixelRawDataError>>;

#endif  // DataFormats_SiPixelRawData_interface_SiPixelFormatterErrors_h
