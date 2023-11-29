#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigisSoAv2_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigisSoAv2_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(SiPixelDigisLayout,
                    SOA_COLUMN(int32_t, clus),
                    SOA_COLUMN(uint32_t, pdigi),
                    SOA_COLUMN(uint32_t, rawIdArr),
                    SOA_COLUMN(uint16_t, adc),
                    SOA_COLUMN(uint16_t, xx),
                    SOA_COLUMN(uint16_t, yy),
                    SOA_COLUMN(uint16_t, moduleId))

using SiPixelDigisSoAv2 = SiPixelDigisLayout<>;
using SiPixelDigisSoAv2View = SiPixelDigisSoAv2::View;
using SiPixelDigisSoAv2ConstView = SiPixelDigisSoAv2::ConstView;

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigisSoAv2_h
