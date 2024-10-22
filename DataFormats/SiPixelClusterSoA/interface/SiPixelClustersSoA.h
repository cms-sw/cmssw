#ifndef DataFormats_SiPixelClusterSoA_SiPixelClustersLayout_h
#define DataFormats_SiPixelClusterSoA_SiPixelClustersLayout_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SiPixelClustersLayout,
                    SOA_COLUMN(uint32_t, moduleStart),
                    SOA_COLUMN(uint32_t, clusInModule),
                    SOA_COLUMN(uint32_t, moduleId),
                    SOA_COLUMN(uint32_t, clusModuleStart))

using SiPixelClustersSoA = SiPixelClustersLayout<>;
using SiPixelClustersSoAView = SiPixelClustersSoA::View;
using SiPixelClustersSoAConstView = SiPixelClustersSoA::ConstView;

#endif  // DataFormats_SiPixelClusterSoA_SiPixelClustersLayout_h
