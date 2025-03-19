#ifndef CondFormats_SiStripObjects_interface_SiStripMappingSoA_h
#define CondFormats_SiStripObjects_interface_SiStripMappingSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

GENERATE_SOA_LAYOUT(SiStripMappingSoALayout,
                    SOA_COLUMN(const uint8_t*, input),
                    SOA_COLUMN(size_t, inoff),
                    SOA_COLUMN(size_t, offset),
                    SOA_COLUMN(uint16_t, length),
                    SOA_COLUMN(uint16_t, fedID),
                    SOA_COLUMN(uint8_t, fedCh),
                    SOA_COLUMN(uint32_t, detID))

using SiStripMappingSoA = SiStripMappingSoALayout<>;
using SiStripMappingView = SiStripMappingSoA::View;
using SiStripMappingConstView = SiStripMappingSoA::ConstView;

#endif  // CondFormats_SiStripObjects_interface_SiStripMappingSoA_h