#ifndef RecoLocalTracker_SiStripClusterizer_interface_SiStripMappingSoA_h
#define RecoLocalTracker_SiStripClusterizer_interface_SiStripMappingSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"

namespace sistrip {
  GENERATE_SOA_LAYOUT(SiStripMappingSoALayout,
                      SOA_COLUMN(const uint8_t*, input),
                      SOA_COLUMN(size_t, inoff),
                      SOA_COLUMN(size_t, offset),
                      SOA_COLUMN(uint16_t, length),
                      //
                      SOA_COLUMN(FEDReadoutMode, readoutMode),
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