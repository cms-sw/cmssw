#include <ostream>

#include "DataFormats/FTLDigiSoA/interface/BTLDigiSoA.h"

namespace btldigi {

  std::ostream& operator<<(std::ostream& out, BTLDigiSoA::View::const_element const& digi) {
    out << "BTL Digi SoA rawId : " << digi.rawId() << ", BC0count = " << digi.BC0count()
        << ", status = " << digi.status() << ", BCcount = " << digi.BCcount() << std::endl
        << "\t sample 0 (left side) : chIDL = " << static_cast<int>(digi.chIDL())
        << ", T1coarseL = " << digi.T1coarseL() << ", T1fineL = " << digi.T1fineL()
        << ", T2coarseL = " << digi.T2coarseL() << ", T2fineL = " << digi.T2fineL()
        << ", EOIcoarseL = " << digi.EOIcoarseL() << ", ChargeL = " << digi.ChargeL() << std::endl
        << "\t sample 1 (right side) : chIDR = " << static_cast<int>(digi.chIDR())
        << ", T1coarseR = " << digi.T1coarseR() << ", T1fineR = " << digi.T1fineR()
        << ", T2coarseR = " << digi.T2coarseR() << ", T2fineR = " << digi.T2fineR()
        << ", EOIcoarseR = " << digi.EOIcoarseR() << ", ChargeR = " << digi.ChargeR();
    return out;
  }
}  // namespace btldigi
