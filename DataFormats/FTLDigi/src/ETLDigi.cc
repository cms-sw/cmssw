#include "DataFormats/FTLDigi/interface/ETLDigi.h"

namespace etldigi {

  std::ostream& operator<<(std::ostream& out, const ETLDigi& digi) {
    out << "ETL Digi SoA rawId : " << digi.krawId() << ", column = " << digi.kcolID() << ", row = " << digi.krowID()
        << ", header = " << digi.kheader() << ", status = " << digi.kstatus() << ", ToA = " << digi.kToAdata()
        << ", ToT = " << digi.kToTdata() << ", CAL = " << digi.kCALdata();
    return out;
  }
}  // namespace etldigi
