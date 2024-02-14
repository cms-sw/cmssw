#ifndef CondFormats_SiPixelObjects_SiPixelMappingHost_h
#define CondFormats_SiPixelObjects_SiPixelMappingHost_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelMappingLayout.h"

using SiPixelMappingHost = PortableHostCollection<SiPixelMappingSoA>;

#endif  // CondFormats_SiPixelObjects_SiPixelMappingHost_h
