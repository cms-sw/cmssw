#ifndef RecoLocalTracker_SiStripClusterizer_interface_SiStripMappingSoA_h
#define RecoLocalTracker_SiStripClusterizer_interface_SiStripMappingSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace sistrip {
  GENERATE_SOA_LAYOUT(SiStripMappingSoALayout,
                      // Detector ID, FED ID and FED channel for indexing
                      SOA_COLUMN(uint32_t, detID),
                      SOA_COLUMN(uint16_t, fedID),
                      SOA_COLUMN(uint8_t, fedCh),
                      // FEDChannel.offset()
                      SOA_COLUMN(uint16_t, fedChOfs),
                      // Offset of the FEDChannel.data() in the flatten raw buffer
                      SOA_COLUMN(uint32_t, fedChDataOfsBuf),
                      // Number of strips in the fedCh
                      SOA_COLUMN(uint32_t, fedChStripsN),
                      // readout mode of the buffer the FED channels are taken
                      SOA_COLUMN(uint8_t, readoutMode),
                      // packet code of the buffer the FED channels are taken
                      SOA_COLUMN(uint8_t, packetCode))

  using SiStripMappingSoA = SiStripMappingSoALayout<>;
  using SiStripMappingView = SiStripMappingSoA::View;
  using SiStripMappingConstView = SiStripMappingSoA::ConstView;
}  // namespace sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_interface_SiStripMappingSoA_h
