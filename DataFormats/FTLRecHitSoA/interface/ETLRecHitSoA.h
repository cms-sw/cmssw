#ifndef DataFormats_FTLRecHitSoA_interface_ETLRecHitSoA_h
#define DataFormats_FTLRecHitSoA_interface_ETLRecHitSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace etlrechit {
  GENERATE_SOA_LAYOUT(ETLRecHitSoALayout,
                      SOA_COLUMN(DetId, detId),
                      SOA_COLUMN(uint8_t, row),
                      SOA_COLUMN(uint8_t, column),
                      SOA_COLUMN(float, toa),
                      SOA_COLUMN(float, tot),
                      SOA_COLUMN(float, toa_error),
                      SOA_COLUMN(uint8_t, flags))

  using ETLRecHitSoA = ETLRecHitSoALayout<>;
  using ETLRecHitSoAView = ETLRecHitSoA::View;
  using ETLRecHitSoAConstView = ETLRecHitSoA::ConstView;

  std::ostream& operator<<(std::ostream& out, ETLRecHitSoA::View::const_element const& etlrh);

}  // namespace etlrechit
#endif  // DataFormats_FTLRecHitSoA_interface_ETLRecHitSoA_h
