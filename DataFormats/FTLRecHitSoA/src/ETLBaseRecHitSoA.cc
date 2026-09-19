#include <ostream>
#include <bitset>
#include <fmt/format.h>

#include "DataFormats/FTLRecHitSoA/interface/ETLBaseRecHitSoA.h"

namespace etlrechit {

  std::ostream& operator<<(std::ostream& out, ETLBaseRecHitSoA::View::const_element const& etlrh) {
    out << "ETL uncalib rechit SoA: "
        << " detID: " << etlrh.detId().rawId() << ", row: " << static_cast<int>(etlrh.row())
        << ", column: " << static_cast<int>(etlrh.column()) << ", toa: " << etlrh.toa() << ", tot: " << etlrh.tot()
        << ", flags: " << std::bitset<2>(etlrh.flags());

    return out;
  }

}  // namespace etlrechit
