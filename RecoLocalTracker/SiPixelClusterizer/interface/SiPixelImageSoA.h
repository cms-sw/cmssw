#ifndef RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageSoA_h
#define RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageSoA_h
#include <array>
#include "DataFormats/SoATemplate/interface/SoALayout.h"
//using SiPixelImage = std::array<std::array<int32_t,160>,416>;
//using SiPixelImage = std::array<std::array<int32_t,162>,418>;
using SiPixelImage = std::array<std::array<int32_t, 162>, 418>;
using SiPixelImageMorph = std::array<std::array<int32_t, 164>, 420>;

GENERATE_SOA_LAYOUT(SiPixelImageLayout,
                    //SOA_COLUMN(std::array<std::array<int,160>,416>, clus))
                    SOA_COLUMN(SiPixelImage, clus))
//SOA_COLUMN(int32_t, clus))

GENERATE_SOA_LAYOUT(SiPixelImageMorphLayout,
                    //SOA_COLUMN(std::array<std::array<int,160>,416>, clus))
                    SOA_COLUMN(SiPixelImageMorph, clus))
//SOA_COLUMN(int32_t, clus))

using SiPixelImageSoA = SiPixelImageLayout<>;
using SiPixelImageSoAView = SiPixelImageSoA::View;
using SiPixelImageSoAConstView = SiPixelImageSoA::ConstView;

using SiPixelImageMorphSoA = SiPixelImageMorphLayout<>;
using SiPixelImageMorphSoAView = SiPixelImageMorphSoA::View;
using SiPixelImageMorphSoAConstView = SiPixelImageMorphSoA::ConstView;

#endif  // RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageDevice_h
