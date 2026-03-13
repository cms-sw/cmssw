#ifndef DataFormats_FTLDigiSoA_interface_BTLDigiSoA_h
#define DataFormats_FTLDigiSoA_interface_BTLDigiSoA_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace btldigi {

  GENERATE_SOA_LAYOUT(BTLDigiSoALayout,
                      SOA_COLUMN(uint32_t, rawId),     // Raw ID of the module/TOFHIR
                      SOA_COLUMN(uint16_t, BC0count),  // BC0 count (reserved)
                      SOA_COLUMN(bool, status),        // status of the TOFHIR
                      SOA_COLUMN(uint32_t, BCcount),
                      SOA_COLUMN(uint8_t, chIDR),       // TOFHIR channel ID, right side of crystal
                      SOA_COLUMN(uint16_t, T1coarseR),  // data from crystal right side
                      SOA_COLUMN(uint16_t, T2coarseR),
                      SOA_COLUMN(uint16_t, EOIcoarseR),
                      SOA_COLUMN(uint16_t, ChargeR),
                      SOA_COLUMN(uint16_t, T1fineR),
                      SOA_COLUMN(uint16_t, T2fineR),
                      SOA_COLUMN(uint16_t, IdleTimeR),
                      SOA_COLUMN(uint8_t, PrevTrigFR),
                      SOA_COLUMN(uint8_t, TACIDR),
                      SOA_COLUMN(uint8_t, chIDL),       // TOFHIR channel ID, left side of crystal
                      SOA_COLUMN(uint16_t, T1coarseL),  // data from crystal left side
                      SOA_COLUMN(uint16_t, T2coarseL),
                      SOA_COLUMN(uint16_t, EOIcoarseL),
                      SOA_COLUMN(uint16_t, ChargeL),
                      SOA_COLUMN(uint16_t, T1fineL),
                      SOA_COLUMN(uint16_t, T2fineL),
                      SOA_COLUMN(uint16_t, IdleTimeL),
                      SOA_COLUMN(uint8_t, PrevTrigFL),
                      SOA_COLUMN(uint8_t, TACIDL))

  using BTLDigiSoA = BTLDigiSoALayout<>;
  using BTLDigiSoAView = BTLDigiSoA::View;
  using BTLDigiSoAConstView = BTLDigiSoA::ConstView;

  std::ostream &operator<<(std::ostream &out, BTLDigiSoA::View::const_element const &digi);

}  // namespace btldigi
#endif  // DataFormats_FTLDigi_interface_BTLDigiSoA_h
