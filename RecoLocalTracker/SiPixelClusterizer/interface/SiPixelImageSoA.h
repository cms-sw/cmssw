#ifndef RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageSoA_h
#define RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageSoA_h

#include <array>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

using SiPixelImage = std::array<std::array<uint16_t, 162>, 418>;
using SiPixelImageMorph = std::array<std::array<uint16_t, 164>, 420>;

GENERATE_SOA_LAYOUT(SiPixelImageLayout, SOA_COLUMN(SiPixelImage, clus))

GENERATE_SOA_LAYOUT(SiPixelImageMorphLayout, SOA_COLUMN(SiPixelImageMorph, clus))

using SiPixelImageSoA = SiPixelImageLayout<>;
using SiPixelImageSoAView = SiPixelImageSoA::View;
using SiPixelImageSoAConstView = SiPixelImageSoA::ConstView;

using SiPixelImageMorphSoA = SiPixelImageMorphLayout<>;
using SiPixelImageMorphSoAView = SiPixelImageMorphSoA::View;
using SiPixelImageMorphSoAConstView = SiPixelImageMorphSoA::ConstView;

#endif  // RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageDevice_h
