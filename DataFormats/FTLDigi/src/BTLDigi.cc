#include "DataFormats/FTLDigi/interface/BTLDigi.h"

namespace btldigi {

  std::ostream& operator<<(std::ostream& out, const BTLDigi& digi) {
    out << "BTL Digi rawId : " << digi.krawId() << ", BC0count = " << digi.kBC0count()
        << ", status = " << digi.kstatus() << ", BCcount = " << digi.kBCcount() << std::endl
        << "\t sample 0 (left side) : chIDMinus = " << static_cast<int>(digi.kchIDMinus())
        << ", T1coarseMinus = " << digi.kT1coarseMinus() << ", T1fineMinus = " << digi.kT1fineMinus()
        << ", T2coarseMinus = " << digi.kT2coarseMinus() << ", T2fineMinus = " << digi.kT2fineMinus()
        << ", EOIcoarseMinus = " << digi.kEOIcoarseMinus() << ", ChargeMinus = " << digi.kChargeMinus() << std::endl
        << "\t sample 1 (right side) : chIDPlus = " << static_cast<int>(digi.kchIDPlus())
        << ", T1coarsePlus = " << digi.kT1coarsePlus() << ", T1finePlus = " << digi.kT1finePlus()
        << ", T2coarsePlus = " << digi.kT2coarsePlus() << ", T2finePlus = " << digi.kT2finePlus()
        << ", EOIcoarsePlus = " << digi.kEOIcoarsePlus() << ", ChargePlus = " << digi.kChargePlus();
    return out;
  }
}  // namespace btldigi
