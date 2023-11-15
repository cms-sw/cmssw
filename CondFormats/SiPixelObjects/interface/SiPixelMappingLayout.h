#ifndef CondFormats_SiPixelObjects_interface_SiPixelMappingLayout_h
#define CondFormats_SiPixelObjects_interface_SiPixelMappingLayout_h

#include <array>
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelROCsStatusAndMapping.h"

GENERATE_SOA_LAYOUT(SiPixelMappingLayout,
                    SOA_COLUMN(unsigned int, fed),
                    SOA_COLUMN(unsigned int, link),
                    SOA_COLUMN(unsigned int, roc),
                    SOA_COLUMN(unsigned int, rawId),
                    SOA_COLUMN(unsigned int, rocInDet),
                    SOA_COLUMN(unsigned int, moduleId),
                    SOA_COLUMN(bool, badRocs),
                    SOA_COLUMN(unsigned char, modToUnpDefault),
                    SOA_SCALAR(unsigned int, size),
                    SOA_SCALAR(bool, hasQuality))

using SiPixelMappingSoA = SiPixelMappingLayout<>;
using SiPixelMappingSoAView = SiPixelMappingSoA::View;
using SiPixelMappingSoAConstView = SiPixelMappingSoA::ConstView;

#endif  // CondFormats_SiPixelObjects_interface_SiPixelMappingLayout_h
