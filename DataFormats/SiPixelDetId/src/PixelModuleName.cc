#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"

std::ostream& operator<<(std::ostream& out, const PixelModuleName::ModuleType& t) {
  switch (t) {
    case (PixelModuleName::v1x2): {
      out << "v1x2";
      break;
    }
    case (PixelModuleName::v1x5): {
      out << "v1x5";
      break;
    }
    case (PixelModuleName::v1x8): {
      out << "v1x8";
      break;
    }
    case (PixelModuleName::v2x3): {
      out << "v2x3";
      break;
    }
    case (PixelModuleName::v2x4): {
      out << "v2x4";
      break;
    }
    case (PixelModuleName::v2x5): {
      out << "v2x5";
      break;
    }
    case (PixelModuleName::v2x8): {
      out << "v2x8";
      break;
    }
    default:
      out << "unknown";
  };
  return out;
}
