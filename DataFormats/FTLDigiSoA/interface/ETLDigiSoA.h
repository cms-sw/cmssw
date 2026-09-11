#ifndef DataFormats_FTLDigiSoA_interface_ETLDigiSoA_h
#define DataFormats_FTLDigiSoA_interface_ETLDigiSoA_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace etldigi {

  GENERATE_SOA_LAYOUT(ETLDigiSoALayout,
                      SOA_COLUMN(uint32_t, rawId),    // Raw ID of the module/ETROC
                      SOA_COLUMN(uint8_t, header),    // Header
                      SOA_COLUMN(uint8_t, status),    // status of the ETROC
                      SOA_COLUMN(uint8_t, colID),     // ETROC column ID
                      SOA_COLUMN(uint8_t, rowID),     // ETROC row ID
                      SOA_COLUMN(uint16_t, ToAdata),  // ToA
                      SOA_COLUMN(uint16_t, ToTdata),  // ToT
                      SOA_COLUMN(uint16_t, CALdata))  // CAL code

  using ETLDigiSoA = ETLDigiSoALayout<>;
  using ETLDigiSoAView = ETLDigiSoA::View;
  using ETLDigiSoAConstView = ETLDigiSoA::ConstView;

  std::ostream &operator<<(std::ostream &out, ETLDigiSoA::View::const_element const &digi);

}  // namespace etldigi
#endif  // DataFormats_FTLDigi_interface_ETLDigiSoA_h
