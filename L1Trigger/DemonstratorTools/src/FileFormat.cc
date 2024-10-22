
#include "L1Trigger/DemonstratorTools/interface/FileFormat.h"

#include <ostream>

namespace l1t::demo {

  std::ostream& operator<<(std::ostream& os, FileFormat format) {
    switch (format) {
      case FileFormat::APx:
        os << "APx";
        break;
      case FileFormat::EMPv1:
        os << "EMPv1";
        break;
      case FileFormat::EMPv2:
        os << "EMPv2";
        break;
      case FileFormat::X2O:
        os << "X2O";
    }
    return os;
  }

}  // namespace l1t::demo
