#include <ostream>
#include <bitset>
#include <fmt/format.h>

#include "DataFormats/FTLRecHitSoA/interface/BTLRecHitSoA.h"

namespace btlrechit {

  std::ostream& operator<<(std::ostream& out, BTLRecHitSoA::View::const_element const& btlrh) {
    out << "BTL rechit SoA: "
        << " detID: " << btlrh.detId().rawId() << ", row: " << static_cast<int>(btlrh.row())
        << ", time1: " << btlrh.time1() << ", time2: " << btlrh.time2() << ", energy: " << btlrh.energy()
        << ", position:	" << btlrh.position() << ", time1 error : " << btlrh.time1_error()
        << ", position error:	" << btlrh.position_error() << ", flags: " << std::bitset<8>(btlrh.flags());
    return out;
  }

}  // namespace btlrechit
