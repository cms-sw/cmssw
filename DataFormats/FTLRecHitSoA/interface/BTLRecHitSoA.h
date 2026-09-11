#ifndef DataFormats_FTLRecHitSoA_interface_BTLRecHitSoA_h
#define DataFormats_FTLRecHitSoA_interface_BTLRecHitSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace btlrechit {
  GENERATE_SOA_LAYOUT(BTLRecHitSoALayout,
                      SOA_COLUMN(DetId, detId),
                      SOA_COLUMN(uint8_t, row),
                      SOA_COLUMN(float, time1),
                      SOA_COLUMN(float, time2),
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, position),
                      SOA_COLUMN(float, time1_error),
                      SOA_COLUMN(float, position_error),
                      SOA_COLUMN(uint8_t, flags))

  using BTLRecHitSoA = BTLRecHitSoALayout<>;
  using BTLRecHitSoAView = BTLRecHitSoA::View;
  using BTLRecHitSoAConstView = BTLRecHitSoA::ConstView;

  std::ostream& operator<<(std::ostream& out, BTLRecHitSoA::View::const_element const& btlrh);

}  // namespace btlrechit
#endif  // DataFormats_FTLRecHitSoA_interface_BTLRecHitSoA_h
