#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsSoA_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"

GENERATE_SOA_LAYOUT(SiPixelDigiErrorsLayout, SOA_COLUMN(SiPixelErrorCompact, pixelErrors), SOA_SCALAR(uint32_t, size))

using SiPixelDigiErrorsSoA = SiPixelDigiErrorsLayout<>;
using SiPixelDigiErrorsSoAView = SiPixelDigiErrorsSoA::View;
using SiPixelDigiErrorsSoAConstView = SiPixelDigiErrorsSoA::ConstView;

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsSoA_h
