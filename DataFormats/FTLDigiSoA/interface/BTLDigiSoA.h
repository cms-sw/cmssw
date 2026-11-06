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
                      SOA_COLUMN(uint8_t, chIDPlus),       // TOFHIR channel ID, plus side of crystal
                      SOA_COLUMN(uint16_t, T1coarsePlus),  // data from crystal plus side
                      SOA_COLUMN(uint16_t, T2coarsePlus),
                      SOA_COLUMN(uint16_t, EOIcoarsePlus),
                      SOA_COLUMN(uint16_t, ChargePlus),
                      SOA_COLUMN(uint16_t, T1finePlus),
                      SOA_COLUMN(uint16_t, T2finePlus),
                      SOA_COLUMN(uint16_t, IdleTimePlus),
                      SOA_COLUMN(uint8_t, PrevTrigFPlus),
                      SOA_COLUMN(uint8_t, TACIDPlus),
                      SOA_COLUMN(uint8_t, chIDMinus),       // TOFHIR channel ID, minus side of crystal
                      SOA_COLUMN(uint16_t, T1coarseMinus),  // data from crystal minus side
                      SOA_COLUMN(uint16_t, T2coarseMinus),
                      SOA_COLUMN(uint16_t, EOIcoarseMinus),
                      SOA_COLUMN(uint16_t, ChargeMinus),
                      SOA_COLUMN(uint16_t, T1fineMinus),
                      SOA_COLUMN(uint16_t, T2fineMinus),
                      SOA_COLUMN(uint16_t, IdleTimeMinus),
                      SOA_COLUMN(uint8_t, PrevTrigFMinus),
                      SOA_COLUMN(uint8_t, TACIDMinus))

  using BTLDigiSoA = BTLDigiSoALayout<>;
  using BTLDigiSoAView = BTLDigiSoA::View;
  using BTLDigiSoAConstView = BTLDigiSoA::ConstView;

  std::ostream &operator<<(std::ostream &out, BTLDigiSoA::View::const_element const &digi);

}  // namespace btldigi
#endif  // DataFormats_FTLDigi_interface_BTLDigiSoA_h
