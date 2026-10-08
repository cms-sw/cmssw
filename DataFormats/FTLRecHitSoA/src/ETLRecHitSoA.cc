#include <ostream>
#include <bitset>
#include <fmt/format.h>

#include "DataFormats/FTLRecHitSoA/interface/ETLRecHitSoA.h"

namespace etlrechit {

  std::ostream& operator<<(std::ostream& out, ETLRecHitSoA::View::const_element const& etlrh) {
    out << "ETL rechit SoA: "
        << " detID: " << etlrh.detId().rawId() << ", row: " << static_cast<int>(etlrh.row())
        << ", column: " << static_cast<int>(etlrh.column()) << ", toa: " << etlrh.toa() << ", tot: " << etlrh.tot()
        << ", toa error : " << etlrh.toa_error() << ", flags: " << std::bitset<8>(etlrh.flags());
    return out;
  }

}  // namespace etlrechit
