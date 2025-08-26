
#ifndef DataFormats_SiStripDigiSoA_interface_SiStripDigiSoA_h
#define DataFormats_SiStripDigiSoA_interface_SiStripDigiSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace sistrip {
  GENERATE_SOA_LAYOUT(SiStripDigiSoALayout,
                      SOA_COLUMN(uint8_t, adc),
                      SOA_COLUMN(uint16_t, channel),
                      SOA_COLUMN(uint16_t, stripId),
                      SOA_SCALAR(uint32_t, nbGoodCandidates),
                      SOA_SCALAR(uint32_t, nbCandidates))

  using SiStripDigiSoA = SiStripDigiSoALayout<>;
  using SiStripDigiView = SiStripDigiSoA::View;
  using SiStripDigiConstView = SiStripDigiSoA::ConstView;
}  // namespace sistrip

#endif
