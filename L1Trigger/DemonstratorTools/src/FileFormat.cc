
#include "L1Trigger/DemonstratorTools/interface/FileFormat.h"

#include <ostream>

namespace l1t::demo {

  std::ostream& operator<<(std::ostream& os, FileFormat format) {
    switch (format) {
      case FileFormat::APx:
        os << "APx";
        break;
      case FileFormat::EMP:
        os << "EMP";
        break;
      case FileFormat::X20:
        os << "X20";
    }
    return os;
  }

}  // namespace l1t::demo
