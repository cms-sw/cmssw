#include <ostream>

#include "DataFormats/FTLDigiSoA/interface/ETLDigiSoA.h"

namespace etldigi {

  std::ostream& operator<<(std::ostream& out, ETLDigiSoA::View::const_element const& digi) {
    out << "ETL Digi SoA rawId : " << digi.rawId() << ", column = " << digi.colID() << ", row = " << digi.rowID()
        << ", header = " << digi.header() << ", status = " << digi.status() << ", ToA = " << digi.ToAdata()
        << ", ToT = " << digi.ToTdata() << ", CAL = " << digi.CALdata();
    return out;
  }
}  // namespace etldigi
