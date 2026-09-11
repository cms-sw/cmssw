#include <ostream>

#include "DataFormats/FTLDigiSoA/interface/BTLDigiSoA.h"

namespace btldigi {

  std::ostream& operator<<(std::ostream& out, BTLDigiSoA::View::const_element const& digi) {
    out << "BTL Digi SoA rawId : " << digi.rawId() << ", BC0count = " << digi.BC0count()
        << ", status = " << digi.status() << ", BCcount = " << digi.BCcount() << std::endl
        << "\t sample 0 (left side) : TACIDMinus = " << static_cast<int>(digi.TACIDMinus())
        << ", T1coarseMinus = " << digi.T1coarseMinus() << ", T1fineMinus = " << digi.T1fineMinus()
        << ", T2coarseMinus = " << digi.T2coarseMinus() << ", T2fineMinus = " << digi.T2fineMinus()
        << ", EOIcoarseMinus = " << digi.EOIcoarseMinus() << ", ChargeMinus = " << digi.ChargeMinus() << std::endl
        << "\t sample 1 (right side) : TACIDPlus = " << static_cast<int>(digi.TACIDPlus())
        << ", T1coarsePlus = " << digi.T1coarsePlus() << ", T1finePlus = " << digi.T1finePlus()
        << ", T2coarsePlus = " << digi.T2coarsePlus() << ", T2finePlus = " << digi.T2finePlus()
        << ", EOIcoarsePlus = " << digi.EOIcoarsePlus() << ", ChargePlus = " << digi.ChargePlus();
    return out;
  }
}  // namespace btldigi
