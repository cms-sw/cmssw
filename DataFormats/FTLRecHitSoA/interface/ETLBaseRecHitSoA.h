#ifndef DataFormats_FTLRecHitSoA_interface_ETLBaseRecHitSoA_h
#define DataFormats_FTLRecHitSoA_interface_ETLBaseRecHitSoA_h

#include <ostream>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace etlrechit {
  GENERATE_SOA_LAYOUT(ETLBaseRecHitSoALayout,
                      SOA_COLUMN(DetId, detId),
                      SOA_COLUMN(uint8_t, row),
                      SOA_COLUMN(uint8_t, column),
                      SOA_COLUMN(float, toa),
                      SOA_COLUMN(float, tot),
                      SOA_COLUMN(uint8_t, flags))

  using ETLBaseRecHitSoA = ETLBaseRecHitSoALayout<>;
  using ETLBaseRecHitSoAView = ETLBaseRecHitSoA::View;
  using ETLBaseRecHitSoAConstView = ETLBaseRecHitSoA::ConstView;

  std::ostream& operator<<(std::ostream& out, ETLBaseRecHitSoA::View::const_element const& etlrh);
}  // namespace etlrechit
#endif  // DataFormats_FTLRecHitSoA_interface_ETLBaseRecHitSoA_h
