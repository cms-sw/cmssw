#include "CondFormats/Calibration/src/headers.h"

namespace CondFormats_Calibration {
  struct dictionary {
    fixedArray<unsigned short, 2097> d;
    std::map<std::string, Algo> e;
    std::pair<std::string, Algo> e1;
    std::map<std::string, Algob> e2;
    std::pair<std::string, Algob> e3;
  };
}  // namespace CondFormats_Calibration
