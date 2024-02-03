#ifndef CondFormats_SiPixelObjects_interface_alpaka_SiPixelMappingDevice_h
#define CondFormats_SiPixelObjects_interface_alpaka_SiPixelMappingDevice_h

#include <cstdint>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelMappingLayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using SiPixelMappingDevice = PortableCollection<SiPixelMappingSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_SiPixelMappingSoA_alpaka_SiPixelClustersDevice_h
