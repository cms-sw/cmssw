#ifndef DataFormats_FTLRecHitSoA_interface_BTLBaseRecHitSoA_h
#define DataFormats_FTLRecHitSoA_interface_BTLBaseRecHitSoA_h

#include <ostream>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace btlrechit {
  GENERATE_SOA_LAYOUT(BTLBaseRecHitSoALayout,
                      SOA_COLUMN(DetId, detId),
                      SOA_COLUMN(uint8_t, row),
                      SOA_COLUMN(float, time1Plus),
                      SOA_COLUMN(float, time2Plus),
                      SOA_COLUMN(float, ampPlus),
                      SOA_COLUMN(uint16_t, idleTimePlus),
                      SOA_COLUMN(uint8_t, flagsPlus),
                      SOA_COLUMN(float, time1Minus),
                      SOA_COLUMN(float, time2Minus),
                      SOA_COLUMN(float, ampMinus),
                      SOA_COLUMN(uint16_t, idleTimeMinus),
                      SOA_COLUMN(uint8_t, flagsMinus))

  using BTLBaseRecHitSoA = BTLBaseRecHitSoALayout<>;
  using BTLBaseRecHitSoAView = BTLBaseRecHitSoA::View;
  using BTLBaseRecHitSoAConstView = BTLBaseRecHitSoA::ConstView;

  std::ostream& operator<<(std::ostream& out, BTLBaseRecHitSoA::View::const_element const& btlrh);
}  // namespace btlrechit
#endif  // DataFormats_FTLRecHitSoA_interface_BTLBaseRecHitSoA_h
