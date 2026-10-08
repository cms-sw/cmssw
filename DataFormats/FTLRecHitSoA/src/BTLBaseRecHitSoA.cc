#include <ostream>
#include <bitset>
#include <fmt/format.h>

#include "DataFormats/FTLRecHitSoA/interface/BTLBaseRecHitSoA.h"

namespace btlrechit {

  std::ostream& operator<<(std::ostream& out, BTLBaseRecHitSoA::View::const_element const& btlrh) {
    out << "BTL uncalib rechit SoA: "
        << " detID: " << btlrh.detId().rawId() << ", row: " << static_cast<int>(btlrh.row())
        << ", time1 Plus: " << btlrh.time1Plus() << ", time2 Plus: " << btlrh.time2Plus()
        << ", amplitude Plus: " << btlrh.ampPlus() << ", idleTime Plus: " << btlrh.idleTimePlus()
        << ", flags Plus: " << std::bitset<2>(btlrh.flagsMinus()) << ", time1 Minus: " << btlrh.time1Minus()
        << ", time2 Minus: " << btlrh.time1Minus() << ", amplitude Minus: " << btlrh.ampMinus()
        << ", idleTime Minus: " << btlrh.idleTimeMinus() << ", flags Minus: " << std::bitset<2>(btlrh.flagsMinus());

    return out;
  }

}  // namespace btlrechit
