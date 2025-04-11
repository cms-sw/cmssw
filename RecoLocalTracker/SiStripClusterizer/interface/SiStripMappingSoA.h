#ifndef RecoLocalTracker_SiStripClusterizer_interface_SiStripMappingSoA_h
#define RecoLocalTracker_SiStripClusterizer_interface_SiStripMappingSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"

namespace sistrip {
  GENERATE_SOA_LAYOUT(SiStripMappingSoALayout,
                      // pointer to FEDChannel data
                      SOA_COLUMN(const uint8_t*, input),
                      // FEDChannel->offset
                      SOA_COLUMN(size_t, inoff),
                      // global offset for the FEDChannel in the rawFEDBuffer
                      SOA_COLUMN(size_t, offset),
                      // FEDChannel->length
                      SOA_COLUMN(uint16_t, length),
                      // readout mode of the buffer the FED channels are taken
                      SOA_COLUMN(FEDReadoutMode, readoutMode),
                      // packet code of the buffer the FED channels are taken
                      SOA_COLUMN(uint8_t, packetCode),
                      //
                      SOA_COLUMN(uint16_t, fedID),
                      SOA_COLUMN(uint8_t, fedCh),
                      SOA_COLUMN(uint32_t, detID))

  using SiStripMappingSoA = SiStripMappingSoALayout<>;
  using SiStripMappingView = SiStripMappingSoA::View;
  using SiStripMappingConstView = SiStripMappingSoA::ConstView;
}  // namespace sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_interface_SiStripMappingSoA_h
